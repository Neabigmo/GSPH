from pathlib import Path

import numpy as np
import pytest

from experiments.external_metal_site import match_sites, read_probes, coordinate_cutoff, reference_qc, chosen_atoms


def test_match_maximizes_cardinality_before_distance():
    pred = [[0.,0.,0.],[.9,0.,0.]]
    refs = [[.1,0.,0.],[-.9,0.,0.]]
    metrics, matches = match_sites(pred,refs,1.)
    assert metrics['tp'] == 2 and metrics['fp'] == metrics['fn'] == 0
    assert {(i,j) for i,j,_ in matches} == {(0,1),(1,0)}


def test_no_double_matching_and_exact_threshold():
    metrics, pairs = match_sites([[0,0,0],[.5,0,0]],[[1,0,0]],1.)
    assert metrics['tp'] == 1 and metrics['fp'] == 1 and metrics['fn'] == 0
    assert pairs[0][2] == .5
    assert metrics['precision'] == .5 and metrics['recall'] == 1
    assert metrics['f1'] == pytest.approx(2/3)
    assert match_sites([[0,0,0]],[[1,0,0]],1.)[0]['tp'] == 1


@pytest.mark.parametrize('pred,refs,counts', [([],[],(0,0,0)),([],[[0,0,0]],(0,0,1)),
                                            ([[0,0,0]],[],(0,1,0)),([[0,0,0]],[[3,0,0]],(0,1,1))])
def test_empty_or_unmatched(pred,refs,counts):
    metrics,pairs = match_sites(pred,refs,1.)
    assert tuple(metrics[k] for k in ['tp','fp','fn']) == counts
    assert not pairs and np.isnan(metrics['tp_mean_error_A'])


@pytest.mark.parametrize('pred,threshold', [([[np.nan,0,0]],1.),([[0,0,0]],0.)])
def test_invalid_matching(pred,threshold):
    with pytest.raises(ValueError):
        match_sites(pred,[[0,0,0]],threshold)


def test_official_probe_parser(tmp_path):
    p=tmp_path/'probes.pdb'
    assert read_probes(p) == []
    p.write_text('HETATM    1 ZN    ZN A  1       1.234  -2.345   3.456  0.51  0.0           ZN2+\n')
    assert read_probes(p) == [dict(x=1.234,y=-2.345,z=3.456,confidence=.51)]
    p.write_text('HETATM broken\n')
    with pytest.raises(IndexError):
        read_probes(p)


def test_predeclared_coordination_radii():
    assert coordinate_cutoff('N') == coordinate_cutoff('O') == 2.6
    assert coordinate_cutoff('S') == 2.8 and coordinate_cutoff('C') == 0


def test_altloc_keeps_a_coherent_highest_occupancy_conformer():
    from types import SimpleNamespace
    def atom(name,label,occ):
        return SimpleNamespace(name=name,altloc=label,occ=occ,element=SimpleNamespace(is_hydrogen=False))
    result=chosen_atoms([atom('N','A',.6),atom('N','B',.4),atom('SG','A',.5),atom('SG','B',.5),atom('CA','\x00',1.)])
    assert result['N'].altloc == result['SG'].altloc == 'A'
    assert result['CA'].altloc == '\x00'
    assert chosen_atoms([]) == {}


@pytest.mark.parametrize('occupancy,b_factor,missing,passed', [(1.,10.,False,True),
                           (.9,10.,False,True),(.8,10.,False,False),(1.,30.,False,False),(1.,10.,True,False)])
def test_reference_coordinate_and_quality_filters(tmp_path,occupancy,b_factor,missing,passed):
    gemmi = pytest.importorskip('gemmi')
    s=gemmi.Structure()
    s.cell=gemmi.UnitCell(50,50,50,90,90,90)
    s.spacegroup_hm='P 1'
    model=gemmi.Model('1')
    chain=gemmi.Chain('A')
    for i,xyz in enumerate([(2.3,0,0),(-2.3,0,0),(0,2.3,0),(0,0,2.3)]):
        r=gemmi.Residue()
        r.name='CYS';r.seqid=gemmi.SeqId(i+1,' ')
        for j,name in enumerate(['N','CA','C','O','CB','SG']):
            if missing and i==0 and name=='O':
                continue
            a=gemmi.Atom();a.name=name;a.element=gemmi.Element('S' if name=='SG' else name[0])
            a.pos=gemmi.Position(*xyz) if name=='SG' else gemmi.Position(15+j,15+i,15)
            a.occ=1.;a.b_iso=10.;r.add_atom(a)
        chain.add_residue(r)
    model.add_chain(chain)
    metals=gemmi.Chain('Z');r=gemmi.Residue();r.name='ZN';r.seqid=gemmi.SeqId(1,' ')
    a=gemmi.Atom();a.name='ZN';a.element=gemmi.Element('Zn');a.pos=gemmi.Position(0,0,0)
    a.occ=occupancy;a.b_iso=b_factor;r.add_atom(a);metals.add_residue(r);model.add_chain(metals)
    s.add_model(model);s.setup_entities()
    p=tmp_path/'test.cif';s.make_mmcif_document().write_file(str(p))
    result=reference_qc(p,'test').iloc[0]
    assert result.n_macro_atoms==4 and result.n_macro_residues==4
    assert result.automatic_QC_pass==passed
    assert not result.full_design_validated


def test_evaluation_writes_headers_for_empty_predictions(tmp_path):
    import json
    from types import SimpleNamespace
    import pandas as pd
    from experiments.external_metal_site import evaluate
    tables=tmp_path/'tables';tables.mkdir()
    output=tmp_path/'metal3d-predictions';output.mkdir()
    (output/'runtime.json').write_text(json.dumps({'test':dict(pdb_id='test',status='success',seconds=1.,n_predictions=0)}))
    pd.DataFrame(columns=['pdb_id','site_id','automatic_QC_pass','x','y','z']).to_csv(tables/'external_metal_site_qc.csv',index=False)
    evaluate(SimpleNamespace(data=tmp_path,tables=tables),['test'])
    assert pd.read_csv(tables/'external_metal_predictions.csv').empty
    summary=pd.read_csv(tables/'external_metal_summary.csv')
    assert summary.n_structures.eq(1).all() and summary.tp.eq(0).all()
    assert summary.f1.isna().all()
