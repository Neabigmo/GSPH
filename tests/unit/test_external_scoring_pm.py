import numpy as np
import pandas as pd
import pytest

from experiments.external_scoring_pm import evaluate, paired_bootstrap


def example():
    return pd.DataFrame(dict(target=['1ABC'] * 4, method=['energy'] * 4,
                             structure=['1ABC.pdb', 'a.pdb', 'b.pdb', 'c.pdb'],
                             score=[100., 1., 2., 3.], rmsd=[0., 2., 4., 6.]))


def test_decoy_only_energy_direction_and_top1():
    result = evaluate(example()).iloc[0]
    assert result.n == 3 and result.n_with_native == 4
    assert result.spearman == 1 and result.top1_rmsd == 2
    assert result.spearman_with_native < 0


@pytest.mark.parametrize('problem', ['empty', 'duplicate', 'nonfinite', 'constant'])
def test_invalid_scores(problem):
    frame = example()
    if problem == 'empty':
        frame = frame.iloc[:0]
    elif problem == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif problem == 'nonfinite':
        frame.loc[0, 'score'] = np.nan
    else:
        frame['score'] = 1.
    with pytest.raises(ValueError):
        evaluate(frame)


def test_paired_target_bootstrap():
    result = paired_bootstrap([.2] * 20)
    assert result['n_boot'] == 2000 and result['n_targets'] == 20
    assert np.isclose(result['mean_diff'], .2)
    assert np.isclose(result['ci_lo'], .2) and np.isclose(result['ci_hi'], .2)
    assert paired_bootstrap([-.2, .1, .4]) == paired_bootstrap([-.2, .1, .4])


@pytest.mark.parametrize('values', [[], [np.nan], [[1., 2.]]])
def test_bad_bootstrap(values):
    with pytest.raises(ValueError):
        paired_bootstrap(values)
