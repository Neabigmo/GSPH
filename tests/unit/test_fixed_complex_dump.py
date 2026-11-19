from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import pandas as pd
import pytest

from experiments.exp05_molecular_landscape.deep_complex_dump import fixed_units
from experiments.exp05_molecular_landscape import deep_complex_dump as dump

ROOT = Path(__file__).resolve().parents[2]


def fixture_inputs(root, task):
    sid = 'target_decoy' if task == 'pd' else 'decoy'
    folder = root / 'Bound_docking/target/relaxed_top_complexes' if task == 'pd' else root / 'target'
    folder.mkdir(parents=True)
    (folder / f'{sid}.pdb').write_text('synthetic')
    if task == 'pd':
        (folder.parent / 'cad_scores.txt').write_text(f'model score site_score\n{sid}.pdb 0.7 0.6\n')
    else:
        (root / 'RMSD').mkdir()
        (root / 'RMSD/target.rmsd').write_text(f'{sid}.pdb 2.0\n')
    manifest = root / 'ids.tsv'
    manifest.write_text(f'task\tgroup\tstructure_id\n{task}\ttarget\t{sid}\n')
    return sid, manifest


@pytest.mark.parametrize('task', ['pd', 'rna'])
def test_fixed_inputs_never_reselect(tmp_path, task):
    sid, manifest = fixture_inputs(tmp_path, task)
    selected = fixed_units(task, tmp_path, manifest)
    assert [u['structure_id'] for u in selected] == [sid]
    assert selected[0]['label_cadscore' if task == 'pd' else 'label_rmsd_official'] is not None


@pytest.mark.parametrize('problem', ['empty', 'duplicate', 'missing', 'group', 'label', 'ambiguous'])
def test_fixed_inputs_reject_inconsistent_pool(tmp_path, problem):
    sid, manifest = fixture_inputs(tmp_path, 'rna')
    if problem == 'empty':
        manifest.write_text('task\tgroup\tstructure_id\n')
    elif problem == 'duplicate':
        manifest.write_text(manifest.read_text() + f'rna\ttarget\t{sid}\n')
    elif problem == 'missing':
        (tmp_path / f'target/{sid}.pdb').unlink()
    elif problem == 'group':
        manifest.write_text(f'task\tgroup\tstructure_id\nrna\twrong\t{sid}\n')
    elif problem == 'label':
        (tmp_path / 'RMSD/target.rmsd').unlink()
    else:
        (tmp_path / 'other').mkdir()
        (tmp_path / f'other/{sid}.pdb').write_text('duplicate')
    with pytest.raises(ValueError):
        fixed_units('rna', tmp_path, manifest)


def test_fixed_manifest_is_exact_archived_cohort():
    fixed = pd.read_csv(ROOT / 'manifests/union_audit_structures.tsv', sep='\t')
    archive = pd.read_parquet(ROOT / 'results/tables/path_landscape.parquet')
    assert len(fixed) == 16
    assert set(zip(fixed.task, fixed.structure_id)) == set(zip(archive.task, archive.structure_id))


@pytest.mark.parametrize('fails', [False, True])
def test_fixed_cli_dispatch_and_failure_status(tmp_path, monkeypatch, fails):
    sid, manifest = fixture_inputs(tmp_path, 'rna')
    units = fixed_units('rna', tmp_path, manifest)
    monkeypatch.setattr(dump, 'fixed_units', lambda *args: units)
    monkeypatch.setattr(dump.DS, 'pick_rna', lambda *args: pytest.fail('Fixed mode must not select again'))
    monkeypatch.setitem(sys.modules, 'intramolecular', SimpleNamespace(
        parse_structure=lambda path: [], extract_intra=lambda *a, **kw: {}, DEFAULT_INTRA_SEP=1))
    monkeypatch.setitem(sys.modules, 'sparse', SimpleNamespace(rna_event_filter=lambda *a, **kw: ({}, {})))
    def export(*args):
        if fails:
            raise RuntimeError('synthetic extraction failure')
        return dict(n_dir_paths=0, n_und_paths=0, n_canon_solved=0)
    monkeypatch.setattr(dump, 'dump_complex', export)
    monkeypatch.setenv('GSPH_H3_CLASH_RATIO', '0.6')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['deep_complex_dump', '--task', 'rna', '--pm-root', str(tmp_path),
                                     '--fixed-ids', str(manifest), '--out', str(out)])
    if fails:
        with pytest.raises(SystemExit) as error:
            dump.main()
        assert error.value.code == 1
        assert 'synthetic extraction failure' in (out / f'{sid}.complex_error.txt').read_text()
    else:
        dump.main()
        record = json.loads((out / f'{sid}.complex_manifest.json').read_text())
        assert record['rna_clash_ratio'] == '0.6'
        assert record['fixed_ids_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert record['input_sha256'] == hashlib.sha256(Path(units[0]['path']).read_bytes()).hexdigest()
