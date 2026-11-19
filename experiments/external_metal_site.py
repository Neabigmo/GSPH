"""Audit Zn references and evaluate official Metal3D predictions without retraining."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


SIDECHAINS = dict(ALA='CB', ARG='CB CG CD NE CZ NH1 NH2', ASN='CB CG OD1 ND2',
                  ASP='CB CG OD1 OD2', CYS='CB SG', GLN='CB CG CD OE1 NE2',
                  GLU='CB CG CD OE1 OE2', GLY='', HIS='CB CG ND1 CD2 CE1 NE2',
                  ILE='CB CG1 CG2 CD1', LEU='CB CG CD1 CD2', LYS='CB CG CD CE NZ',
                  MET='CB CG SD CE', PHE='CB CG CD1 CD2 CE1 CE2 CZ', PRO='CB CG CD',
                  SER='CB OG', THR='CB OG1 CG2', TRP='CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2',
                  TYR='CB CG CD1 CD2 CE1 CE2 CZ OH', VAL='CB CG1 CG2', MSE='CB CG SE CE')
COFACTORS = {'HEM', 'HEC', 'HEA', 'HEO', 'HEB', 'HEV', 'FES', 'SF4', 'F3S',
             'CHL', 'CLA', 'CHB', 'B12', 'COB', 'CNC'}
EXTERNAL_QC = 'MetalPDB_site_mapping;BioLiP2_biological_relevance;CheckMyMetal_valence_vecsum_geometry;unmodelled_entire_shell_residues'


def chosen_atoms(residue):
    conformers = {}
    for atom in residue:
        if atom.altloc not in {'\x00', ' '} and not atom.element.is_hydrogen:
            conformers.setdefault(atom.altloc, []).append(atom.occ)
    preferred = max(conformers, key=lambda label: (np.mean(conformers[label]), label)) if conformers else None
    chosen = {}
    for atom in residue:
        if atom.element.is_hydrogen or (preferred and atom.altloc not in {'\x00', ' ', preferred}):
            continue
        previous = chosen.get(atom.name)
        if previous is None or (atom.occ, atom.altloc) > (previous.occ, previous.altloc):
            chosen[atom.name] = atom
    return chosen


def coordinate_cutoff(element):
    return {'N': 2.6, 'O': 2.6, 'S': 2.8}.get(element, 0.)


def reference_qc(cif_path, pdb_id):
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    structure.setup_cell_images()
    model = structure[0]
    selected = {}
    raw = []
    for ci, chain in enumerate(model):
        for ri, residue in enumerate(chain):
            atoms = chosen_atoms(residue)
            selected[ci, ri] = atoms
            raw.extend((ci, ri, a) for a in atoms.values() if a.element.name == 'Zn')
    search = gemmi.NeighborSearch(structure, 3.).populate(include_h=False)
    rows = []
    chains_with_incomplete_shell = set()
    for ci, ri, ion in raw:
        chain, residue = model[ci], model[ci][ri]
        ligands, sym_contacts, seen = [], [], set()
        for mark in search.find_neighbors(ion, min_dist=.1, max_dist=2.8):
            cra = mark.to_cra(model)
            cutoff = coordinate_cutoff(cra.atom.element.name)
            if not cutoff:
                continue
            selected_atom = selected[mark.chain_idx, mark.residue_idx].get(cra.atom.name)
            if selected_atom is None or selected_atom.altloc != cra.atom.altloc or cra.atom.occ <= 0:
                continue
            nearest = structure.cell.find_nearest_pbc_position(ion.pos, cra.atom.pos, mark.image_idx)
            distance = nearest.dist(ion.pos)
            if distance > cutoff:
                continue
            key = (mark.chain_idx, mark.residue_idx, cra.atom.name,
                   tuple(round(v, 5) for v in (nearest.x, nearest.y, nearest.z)))
            if key in seen:
                continue
            seen.add(key)
            macro = cra.residue.entity_type == gemmi.EntityType.Polymer
            symmetric = nearest.dist(cra.atom.pos) > 1e-4
            record = dict(chain=cra.chain.name, residue=str(cra.residue.seqid),
                          resname=cra.residue.name, atom=cra.atom.name, element=cra.atom.element.name,
                          distance=distance, b_iso=cra.atom.b_iso, macromolecular=macro,
                          symmetry_copy=symmetric)
            ligands.append(record)
            if symmetric:
                sym_contacts.append(record)
            if macro and not symmetric and cra.residue.name in SIDECHAINS:
                expected = set(('N CA C O ' + SIDECHAINS[cra.residue.name]).split())
                actual = {name for name, atom in selected[mark.chain_idx, mark.residue_idx].items() if atom.occ > 0}
                if expected - actual:
                    chains_with_incomplete_shell.add(cra.chain.name)
        macro = [a for a in ligands if a['macromolecular'] and not a['symmetry_copy']]
        ligand_mean_b = float(np.mean([a['b_iso'] for a in ligands])) if ligands else np.nan
        macro_residues = {(a['chain'], a['residue']) for a in macro}
        reasons = []
        if ion.occ < .9 - 1e-6:
            reasons.append('occupancy_lt_0.9')
        if not np.isfinite(ligand_mean_b) or ion.b_iso > 2 * ligand_mean_b + 1e-6:
            reasons.append('B_gt_2x_ligand_mean_or_no_ligands')
        if len(macro) < 3 or len(macro_residues) < 2:
            reasons.append('fewer_than_3_macro_atoms_or_2_residues')
        if sym_contacts:
            reasons.append('symmetry_coordination')
        if any(a['resname'] in COFACTORS for a in ligands):
            reasons.append('known_geometry_determining_cofactor')
        rows.append(dict(pdb_id=pdb_id, site_id=f'{chain.name}:{residue.seqid}:{ion.name}',
                         x=ion.pos.x, y=ion.pos.y, z=ion.pos.z, occupancy=ion.occ,
                         b_iso=ion.b_iso, ligand_mean_b=ligand_mean_b,
                         n_macro_atoms=len(macro), n_macro_residues=len(macro_residues),
                         n_symmetry_ligands=len(sym_contacts),
                         macro_chains=';'.join(sorted({a['chain'] for a in macro})),
                         incomplete_shell_chains='', exclusion_reasons=';'.join(reasons),
                         ligands_json=json.dumps(ligands, separators=(',', ':')),
                         unverified_conditions=EXTERNAL_QC,
                         automatic_QC_pass=False, full_design_validated=False))
    for row in rows:
        affected = set(row['macro_chains'].split(';')) & chains_with_incomplete_shell
        row['incomplete_shell_chains'] = ';'.join(sorted(affected))
        if affected:
            row['exclusion_reasons'] = ';'.join(filter(None, [row['exclusion_reasons'], 'chain_with_incomplete_modelled_shell_residue']))
        row['automatic_QC_pass'] = not row['exclusion_reasons']
    return pd.DataFrame(rows)


def match_sites(predictions, references, threshold):
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 3)
    references = np.asarray(references, dtype=float).reshape(-1, 3)
    if threshold <= 0 or not np.isfinite(predictions).all() or not np.isfinite(references).all():
        raise ValueError('Positive threshold and finite coordinates required')
    n, m = len(predictions), len(references)
    distances = cdist(predictions, references)
    matches = []
    if n and m:
        unmatched_cost = (max(n, m) + 1) * (threshold + 1)
        cost = np.full((n, m + n), unmatched_cost)
        cost[:, :m] = np.where(distances <= threshold, distances, unmatched_cost * (n + m + 2))
        rr, cc = linear_sum_assignment(cost)
        matches = [(int(i), int(j), float(distances[i, j])) for i, j in zip(rr, cc)
                   if j < m and distances[i, j] <= threshold]
    tp = len(matches)
    return dict(tp=tp, fp=n-tp, fn=m-tp,
                precision=tp/n if n else np.nan, recall=tp/m if m else np.nan,
                f1=2*tp/(n+m) if n+m else np.nan,
                tp_mean_error_A=float(np.mean([v[2] for v in matches])) if tp else np.nan,
                tp_median_error_A=float(np.median([v[2] for v in matches])) if tp else np.nan), matches


def read_probes(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('HETATM'):
                fields = line.split()
                result.append(dict(x=float(fields[6]), y=float(fields[7]), z=float(fields[8]),
                                   confidence=float(fields[9])))
    return result


def infer(args, eligible):
    destination = args.data / 'metal3d-predictions'
    destination.mkdir(exist_ok=True)
    ledger = destination / 'runtime.json'
    completed = json.loads(ledger.read_text()) if ledger.exists() else {}
    env = {**os.environ, 'OMP_NUM_THREADS':'8', 'OPENBLAS_NUM_THREADS':'8', 'MKL_NUM_THREADS':'8',
           'PYTHONPATH':str(args.data/'python-headers/usr/lib/python3.10')+':'+str(args.data/'python-headers/usr/lib/python3.10/lib-dynload'),
           'LD_LIBRARY_PATH':str(args.data/'python-headers/usr/lib')+':'+str(args.data/'python-headers/usr/lib/x86_64-linux-gnu')}
    for identifier in eligible:
        if identifier in completed and completed[identifier]['status'] == 'success':
            continue
        pdb = args.data / 'metal-structures' / f'{identifier}.pdb'
        probes = destination / f'{identifier}_probes.pdb'
        log = destination / f'{identifier}.log'
        start = time.perf_counter()
        command = [str(args.python), 'metal3d.py', '--pdb', str(pdb), '--metalbinding',
                   '--softexit', '--writeprobes', '--probefile', str(probes), '--batch-size', '1',
                   '--threshold', '7', '--pthreshold', '0.10']
        with log.open('w') as out:
            try:
                p = subprocess.run(command, cwd=args.metal_repo/'Metal3D', env=env,
                                   stdout=out, stderr=subprocess.STDOUT, timeout=7200)
                status = 'success' if p.returncode == 0 and (probes.exists() or 'no points available for clustering' in log.read_text()) else 'error'
                returncode = p.returncode
            except subprocess.TimeoutExpired:
                status, returncode = 'timeout', None
        completed[identifier] = dict(pdb_id=identifier, seconds=time.perf_counter()-start,
                                     returncode=returncode, status=status,
                                     n_predictions=len(read_probes(probes)) if status == 'success' else None)
        ledger.write_text(json.dumps(completed, indent=2))
        print(json.dumps(completed[identifier]), flush=True)
        if status != 'success':
            raise RuntimeError(f'{identifier} failed; inspect {log}')
    return completed


def prepare(args):
    import gemmi

    audit = pd.read_csv(args.tables/'external_metal_structure_audit.csv')
    eligible = sorted(audit[audit.cohort.eq('zn') & audit.structure_numeric_filters_pass].pdb_id)
    if len(eligible) != 32:
        raise ValueError('Expected exactly 32 structurally eligible Zn structures')
    references = []
    for identifier in eligible:
        cif = args.data/'metal-structures'/f'{identifier}.cif.gz'
        structure = gemmi.read_structure(str(cif))
        if len(structure) > 1:
            del structure[1:]
        structure.write_pdb(str(args.data/'metal-structures'/f'{identifier}.pdb'))
        references.append(reference_qc(cif, identifier))
    pd.concat(references, ignore_index=True).to_csv(args.tables/'external_metal_site_qc.csv', index=False)
    print('prepared', len(eligible), 'structures;', sum(len(r) for r in references), 'Zn sites', flush=True)
    return eligible


def evaluate(args, eligible):
    ledger = json.loads((args.data/'metal3d-predictions/runtime.json').read_text())
    if set(ledger) != set(eligible) or any(v['status'] != 'success' for v in ledger.values()):
        raise ValueError('All 32 structures must finish before evaluation')
    refs = pd.read_csv(args.tables/'external_metal_site_qc.csv')
    predictions, per_structure, matched = [], [], []
    for identifier in eligible:
        pred = read_probes(args.data/'metal3d-predictions'/f'{identifier}_probes.pdb')
        predictions.extend(dict(pdb_id=identifier, prediction_index=i, **p) for i,p in enumerate(pred))
        kept = refs[refs.pdb_id.eq(identifier) & refs.automatic_QC_pass].reset_index(drop=True)
        points = np.array([[p[k] for k in ('x','y','z')] for p in pred]).reshape(-1,3)
        for threshold in [1.,2.]:
            metrics, pairs = match_sites(points, kept[['x','y','z']].to_numpy(), threshold)
            per_structure.append(dict(method='Metal3D',pdb_id=identifier,cohort='zn',threshold_A=threshold,
                                      scope='automatic_QC_partial_design',n_predictions=len(pred),n_references=len(kept),**metrics))
            matched.extend(dict(pdb_id=identifier,threshold_A=threshold,prediction_index=i,
                                reference_site=kept.iloc[j].site_id,distance_A=d) for i,j,d in pairs)
    pd.DataFrame(predictions,columns=['pdb_id','prediction_index','x','y','z','confidence']).to_csv(args.tables/'external_metal_predictions.csv', index=False)
    per_structure = pd.DataFrame(per_structure)
    per_structure.to_csv(args.tables/'external_metal_per_structure.csv', index=False)
    pd.DataFrame(matched, columns=['pdb_id','threshold_A','prediction_index','reference_site','distance_A']).to_csv(args.tables/'external_metal_matches.csv', index=False)
    summary = []
    for threshold, group in per_structure.groupby('threshold_A'):
        tp,fp,fn = group[['tp','fp','fn']].sum()
        errors = [r['distance_A'] for r in matched if r['threshold_A'] == threshold]
        summary.append(dict(method='Metal3D',cohort='zn',threshold_A=threshold,
                            scope='automatic_QC_partial_design',n_structures=len(eligible),
                            n_references=int(tp+fn),tp=int(tp),fp=int(fp),fn=int(fn),
                            precision=tp/(tp+fp) if tp+fp else np.nan,
                            recall=tp/(tp+fn) if tp+fn else np.nan,
                            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                            tp_mean_error_A=float(np.mean(errors)) if errors else np.nan,
                            tp_median_error_A=float(np.median(errors)) if errors else np.nan))
    pd.DataFrame(summary).to_csv(args.tables/'external_metal_summary.csv', index=False)
    pd.DataFrame(ledger.values()).to_csv(args.tables/'external_metal_runtime.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--metal-repo',type=Path)
    parser.add_argument('--python',type=Path)
    parser.add_argument('--action',choices=['prepare','infer','evaluate'],required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        audit = pd.read_csv(args.tables/'external_metal_structure_audit.csv')
        eligible = sorted(audit[audit.cohort.eq('zn') & audit.structure_numeric_filters_pass].pdb_id)
        if args.action == 'infer':
            infer(args,eligible)
        else:
            evaluate(args,eligible)


if __name__ == '__main__':
    main()
