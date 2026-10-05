import numpy as np
import pytest
from gold_forecast.metrics import interval_score, qlike, moving_block_ci


def test_hand_metrics():
    np.testing.assert_allclose(interval_score(np.array([0.,2.,5.]), np.array([1.,1.,1.]), np.array([3.,3.,3.]), .2), [12,2,22])
    np.testing.assert_allclose(qlike([2,0], [4,1]), [np.log(4)+1,0])
    with pytest.raises(ValueError): qlike([1], [0])
    with pytest.raises(ValueError): qlike([1], [np.nan])


def test_paired_block_bootstrap():
    np.testing.assert_allclose(moving_block_ci(np.ones(50)*2, 20, 1), [2,2])
    d = np.arange(10.)
    np.testing.assert_allclose(moving_block_ci(d, 10, 1), [4.5,4.5])
    np.testing.assert_array_equal(moving_block_ci(d, 3, 10), moving_block_ci(d, 3, 10))


def test_short_bootstrap_series_is_flagged():
    import pandas as pd
    from gold_forecast.metrics import paired_analysis
    rows = []
    for day in range(5):
        for i, model in enumerate(['Naive','ARIMA','GARCH']):
            rows.append(dict(origin=f'2024-01-{day+1:02d}', horizon=1, model=model, fit_status='ok',
                             actual_price=100, point_price=100+i, lower80=98, upper80=102,
                             lower95=97, upper95=103))
    result = paired_analysis(pd.DataFrame(rows), dict(seed=1,bootstrap_blocks=[20],bootstrap_samples=50))
    assert result.inference_status.eq('limited_sample').all()
    assert result.effective_block.eq(5).all()
