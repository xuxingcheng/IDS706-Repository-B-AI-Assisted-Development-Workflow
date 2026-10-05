import json
import pandas as pd
import pytest
from gold_forecast import projection
from gold_forecast.models import Forecast


def test_published_calendar():
    dates = projection.projection_dates('2025-08-14')
    assert dates[0] == pd.Timestamp('2025-08-15')
    assert dates[-1] == pd.Timestamp('2026-12-31')
    assert (dates.dayofweek < 5).all()
    assert not dates.isin(pd.to_datetime(projection.HOLIDAYS)).any()
    assert (dates.year == 2026).sum() == 251
    assert pd.Timestamp('2025-11-28') in dates  # Early close remains a session.
    assert pd.Timestamp('2026-12-24') in dates
    with pytest.raises(ValueError):
        projection.projection_dates('2025-08-13')


def test_projection_is_separate_and_all_future_values_are_forecasts(tmp_path, monkeypatch):
    prices = pd.Series([190., 310.], index=pd.to_datetime(['2024-01-02', '2025-08-14']))
    calls = []
    def fake(model, history, horizon, seed, simulations, order):
        pd.testing.assert_series_equal(history, prices)
        calls.append((model, horizon, seed, simulations, order))
        values = {h: dict(point_price=310+h*.1, lower80=290., upper80=380., lower95=270., upper95=450.,
                          return_variance_percent2=1., return_variance_decimal2=.0001)
                  for h in range(1, horizon+1)}
        return Forecast(values, 'ok', {'seed': seed})
    monkeypatch.setattr(projection, 'forecast', fake)
    original = tmp_path / 'test_forecasts.csv'
    original.write_text('do not replace evaluation artifacts')
    spec = dict(order=[3,1], config=dict(seed=706, simulations=10000))
    frame = projection.generate_projection(prices, spec, tmp_path)
    assert original.read_text() == 'do not replace evaluation artifacts'
    assert frame.actual_price.isna().all()
    assert frame.phase.eq('projection_2026').all()
    assert set(frame.model) == {'ARIMA', 'GARCH'}
    assert all(c[3] == 10000 and c[4] == [3,1] for c in calls)
    assert frame.groupby('model').horizon.max().eq(len(projection.projection_dates('2025-08-14'))).all()
    metadata = json.loads((tmp_path/'projection_2026/metadata.json').read_text())
    assert metadata['sessions_in_2026'] == 251
    assert 'not validated' in metadata['note']
    assert 'data:image/png;base64,' in projection.extension_section(tmp_path)
