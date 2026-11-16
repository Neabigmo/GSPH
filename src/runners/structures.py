import os
import tempfile


def _contact_stats(first_rows, second_rows, cutoff2=25.0):
    """Return contact count and touched-atom indices using a spatial index."""
    if not first_rows or not second_rows:
        return 0, set(), set()
    try:
        import numpy as np
        from scipy.spatial import cKDTree
        first_xyz = np.asarray([row[4] for row in first_rows], dtype=float)
        second_xyz = np.asarray([row[4] for row in second_rows], dtype=float)
        tree = cKDTree(second_xyz)
        neighbours = tree.query_ball_point(first_xyz, cutoff2 ** 0.5)
        touched_first = {i for i, hits in enumerate(neighbours) if hits}
        touched_second = {j for hits in neighbours for j in hits}
        return sum(len(hits) for hits in neighbours), touched_first, touched_second
    except ImportError:
        count = 0
        touched_first = set()
        touched_second = set()
        for ia, row_a in enumerate(first_rows):
            xa, ya, za = row_a[4]
            for ib, row_b in enumerate(second_rows):
                xb, yb, zb = row_b[4]
                if (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2 <= cutoff2:
                    count += 1
                    touched_first.add(ia)
                    touched_second.add(ib)
        return count, touched_first, touched_second


def _is_hydrogen(atom):
    element = (getattr(atom, "element", "") or "").strip().upper()
    return element == "H" or atom.get_id().upper().startswith("H")


def parse_structure(path):
    """Return a list of (chain_id, atom_name, residue_id, element, xyz)."""
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError:
        return []
    parser = MMCIFParser(QUIET=True) if path.lower().endswith(".cif") else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("candidate", path)
    except Exception:
        return []
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if _is_hydrogen(atom):
                        continue
                    xyz = tuple(float(v) for v in atom.get_coord())
                    atoms.append((str(chain.id), atom.get_name(), residue.id[1],
                                   (getattr(atom, "element", "") or "").strip().upper(), xyz))
        break
    return atoms


def chain_pair(atoms):
    chains = sorted(set(row[0] for row in atoms))
    if len(chains) < 2:
        return None, None
    # For complexes with more than two chains, choose the pair with the
    # largest heavy-atom contact count rather than assuming chain order.
    by_chain = {chain: [row for row in atoms if row[0] == chain] for chain in chains}
    best = (chains[0], chains[1], -1)
    for index, first in enumerate(chains):
        for second in chains[index + 1:]:
            count, _, _ = _contact_stats(by_chain[first], by_chain[second])
            if count > best[2]:
                best = (first, second, count)
    return best[0], best[1]


def interface_features(path, cutoff=5.0):
    atoms = parse_structure(path)
    chain_a, chain_b = chain_pair(atoms)
    if not chain_a or not chain_b:
        return {"contact_count": None, "interface_area_proxy": None, "chain_a": chain_a or "", "chain_b": chain_b or ""}
    aa = [x for x in atoms if x[0] == chain_a]
    bb = [x for x in atoms if x[0] == chain_b]
    contacts, near_a, near_b = _contact_stats(aa, bb, float(cutoff) ** 2)
    # A reproducible geometric proxy, explicitly not a solvent-accessible BSA.
    area_proxy = float(len(near_a) + len(near_b)) * 20.0
    return {
        "contact_count": contacts,
        "interface_area_proxy": area_proxy,
        "chain_a": chain_a,
        "chain_b": chain_b,
    }


def buried_surface_area(path, chain_a=None, chain_b=None):
    """Calculate BSA from FreeSASA for the first two chains in a PDB file.

    The returned value is the conventional half delta-SASA:
    (SASA(chain A) + SASA(chain B) - SASA(complex)) / 2.
    This adapter intentionally supports PDB input only; mmCIF is left for a
    future parser because silently approximating it would change the baseline.
    """
    if not path.lower().endswith((".pdb", ".ent")):
        raise ValueError("true BSA adapter currently requires PDB input")
    try:
        import freesasa
    except ImportError as exc:
        raise RuntimeError("freesasa is not installed") from exc
    try:
        freesasa.setVerbosity(freesasa.silent)
    except AttributeError:
        pass
    atoms_by_chain = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            chain = line[21:22].strip() or "_"
            atoms_by_chain.setdefault(chain, []).append(line)
    chains = sorted(atoms_by_chain)
    if chain_a is None or chain_b is None:
        if len(chains) < 2:
            raise ValueError("structure has fewer than two chains")
        chain_a, chain_b = chains[:2]
    if chain_a not in atoms_by_chain or chain_b not in atoms_by_chain:
        raise ValueError("requested chains are absent from structure")

    def sasa(lines):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.writelines(lines)
            temp_path = tmp.name
        try:
            return float(freesasa.calc(freesasa.Structure(temp_path)).totalArea())
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # Compare the same selected pair in complex form. Extra chains (solvent,
    # cofactors, or additional copies) must not dilute the pairwise interface.
    complex_sasa = sasa(atoms_by_chain[chain_a] + atoms_by_chain[chain_b])
    a_sasa = sasa(atoms_by_chain[chain_a])
    b_sasa = sasa(atoms_by_chain[chain_b])
    return max(0.0, (a_sasa + b_sasa - complex_sasa) / 2.0)
