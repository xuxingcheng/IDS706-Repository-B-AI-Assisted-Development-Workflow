import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gold_forecast.backtest import rolling, tune, evaluate, final_forecasts, save_json
from gold_forecast.cli import initialize
from gold_forecast.metrics import summarize, common_pairs
from gold_forecast.report import generate


def fixture_data():
    rng = np.random.default_rng(14)
    r, v, last = [], 1., 0.
    for _ in range(280):
        v = .05 + .1*last**2 + .85*v
        last = np.sqrt(v)*rng.normal()
        r.append(last)
    return pd.Series(100*np.exp(np.cumsum(r)/100), index=pd.bdate_range('2020-01-01', periods=280))


def test_end_to_end(tmp_path):
    prices = fixture_data()
    c = dict(train_end=str(prices.index[229].date()), validation_end=str(prices.index[254].date()),
             horizons=[1,5,20], orders=[0], simulations=10000, seed=8,
             bootstrap_blocks=[20,10,40], bootstrap_samples=100)
    d = tmp_path/'smoke'
    initialize(c, d, dict(sha256='fixture', end=str(prices.index[-1].date()), split_counts=[230,25,25]))
    spec = tune(prices, c, d)
    test = evaluate(prices, spec, d)
    final_forecasts(prices, spec, d)
    scores = generate(d)
    assert set(scores.model) == {'Naive','ARIMA','GARCH'}
    assert scores[scores.horizon == 1].attempted.tolist() == [25]*3
    assert scores[scores.horizon == 5].attempted.tolist() == [21]*3
    assert scores[scores.horizon == 20].attempted.tolist() == [6]*3
    assert (test.training_end == test.origin).all()
    assert (test.origin < test.target_date).all()
    for row in test.itertuples():
        assert prices.index.get_loc(row.target_date)-prices.index.get_loc(row.origin) == row.horizon
    html = (d/'comparison.html').read_text()
    assert html.count('data:image/png;base64,') >= 8
    assert 'https://' not in html
    exported = pd.read_csv(d/'comparison_metrics.csv')
    np.testing.assert_allclose(exported.mae, scores.mae)
    forecasts = pd.read_csv(d/'comparison_forecasts.csv')
    assert forecasts[forecasts.phase == 'final'].actual_price.isna().all()
    for name in ['manifest.json','quality_report.json','frozen.json','tuning_scores.csv','validation_forecasts.csv',
                 'test_forecasts.csv','final_forecasts.csv','scores.csv','paired_bootstrap.csv','report.html']:
        assert (d/name).exists()
    with pytest.raises(ValueError): evaluate(prices, spec, d)
    with pytest.raises(ValueError): tune(prices, c, d)
    failed = test.copy()
    failed.loc[0, 'fit_status'] = 'failed'
    s = summarize(failed)
    assert s.failures.sum() == 1
    assert s[s.horizon == failed.loc[0,'horizon']].n.nunique() == 1
    one = test[(test.model == 'Naive') & (test.horizon == 1)]
    naive_score = scores[(scores.model == 'Naive') & (scores.horizon == 1)].iloc[0]
    assert naive_score.mae == pytest.approx(abs(one.actual_price-one.point_price).mean())
    # A rerun on the same frozen settings reproduces all forecast numbers exactly.
    repeated = rolling(prices, 255, 280, c, 'test', d.name, spec['order'])
    pd.testing.assert_frame_equal(test, repeated)


def test_tuning_cannot_access_future(monkeypatch, tmp_path):
    from gold_forecast import backtest
    prices = fixture_data()
    c = dict(train_end=str(prices.index[229].date()), validation_end=str(prices.index[254].date()),
             horizons=[1], orders=[0], simulations=10000, seed=8)
    seen = []
    original = backtest.forecast
    def spy(model, history, *args, **kwargs):
        seen.append(history.index[-1])
        assert history.index[-1] < prices.index[254]
        return original(model, history, *args, **kwargs)
    monkeypatch.setattr(backtest, 'forecast', spy)
    changed = prices.copy()
    changed.iloc[255:] *= 100
    tmp_path.mkdir(exist_ok=True)
    tune(changed, c, tmp_path)
    assert seen


def test_no_common_valid_pairs_is_explicit():
    rows = []
    for model in ['Naive','ARIMA','GARCH']:
        rows.append(dict(model=model, horizon=1, origin='2024-01-01', actual_price=100,
                         fit_status='failed', point_price=np.nan))
    scores = summarize(pd.DataFrame(rows))
    assert scores.failures.sum() == 3
    assert scores.n.sum() == 0
    assert scores.mae.isna().all()
    from gold_forecast.report import interpretation
    from gold_forecast.metrics import paired_analysis
    paired = paired_analysis(pd.DataFrame(rows), {})
    assert 'No valid common' in interpretation(scores, paired)


def test_models_receive_identical_histories(monkeypatch):
    from gold_forecast import backtest
    from gold_forecast.models import Forecast
    histories = {}
    def spy(model, history, horizon, seed, simulations, order):
        histories.setdefault(history.index[-1], []).append((model, history.copy()))
        return Forecast({}, 'failed', {'error': 'injected failure'})
    monkeypatch.setattr(backtest, 'forecast', spy)
    prices = fixture_data()
    rows = rolling(prices, 270, 280, dict(horizons=[1,5,20],seed=8,simulations=10000), 'test','spy',(0,0))
    assert rows.fit_status.eq('failed').all()
    assert 'point_price' in rows and rows.point_price.isna().all()
    for values in histories.values():
        assert [x[0] for x in values] == ['Naive','ARIMA','GARCH']
        for _, history in values[1:]:
            pd.testing.assert_series_equal(history, values[0][1])
