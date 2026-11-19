import pandas as pd

from experiments.external_scoring_pd import evaluate


def test_higher_score_is_better_and_topk_takes_best_label():
    scores = pd.DataFrame(dict(task='pd_bound', method='VoroMQA', group='1ABC',
                               structure_id=['a', 'b', 'c', 'd'], label_cadscore=[.1, .2, .3, .4],
                               score=[1., 2., 3., 4.]))
    row = evaluate(scores).iloc[0]
    assert row.n == 4 and row.spearman == 1
    assert row.top1_label == .4 and row.top5_label == .4


def test_ties_break_by_model_id():
    scores = pd.DataFrame(dict(task='pd_bound', method='rank', group='1ABC',
                               structure_id=['b', 'a', 'c'], label_cadscore=[.9, .1, .5], score=[1., 1., 0.]))
    assert evaluate(scores).iloc[0].top1_label == .1
