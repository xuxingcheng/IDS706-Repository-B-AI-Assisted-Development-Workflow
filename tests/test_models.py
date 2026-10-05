import numpy as np
import pandas as pd
import pytest
from gold_forecast.models import forecast, fit_model, arima_distribution
from gold_forecast.data import returns


@pytest.fixture
def prices():
    rng = np.random.default_rng(81)
    r = rng.normal(0, .9, 350)
    return pd.Series(100*np.exp(np.cumsum(r)/100), index=pd.bdate_range('2019-01-01', periods=350))


@pytest.mark.parametrize('model', ['Naive', 'ARIMA', 'GARCH'])
def test_forecast_contract_and_reproducibility(prices, model):
    a = forecast(model, prices, 20, 15)
    b = forecast(model, prices, 20, 15)
    assert a.status == b.status
    if a.status == 'failed':
        assert model == 'GARCH' and 'nonstationary' in a.details['error']
        assert not a.values  # Boundary fits must be surfaced, never replaced.
        return
    assert a.values == b.values
    for value in a.values.values():
        assert 0 < value['lower95'] <= value['lower80'] <= value['point_price'] <= value['upper80'] <= value['upper95']
        assert value['return_variance_percent2']/10000 == value['return_variance_decimal2']
        if model == 'Naive': assert value['point_price'] == prices.iloc[-1]


def test_arima_joint_covariance(prices):
    fit = fit_model('ARIMA', returns(prices), (1,1))
    mean, cov = arima_distribution(fit, 5)
    np.testing.assert_allclose(np.diag(cov), fit.get_forecast(5).var_pred_mean)
    np.testing.assert_allclose(mean, fit.forecast(5))
    # Independent library conditional simulation also checks off-diagonal covariance.
    sim = np.asarray(fit.simulate(5, anchor='end', repetitions=30000, random_state=50)).reshape(5, -1)
    np.testing.assert_allclose(np.cov(sim), cov, atol=.035)


def test_garch_variance_library_and_recursion():
    rng = np.random.default_rng(90)
    r, v, last = [], 1., 0.
    for _ in range(1200):
        v = .05 + .1*last**2 + .85*v
        last = np.sqrt(v)*rng.normal()
        r.append(last)
    p = pd.Series(100*np.exp(np.r_[0, np.cumsum(r)]/100))
    f = forecast('GARCH', p, 20, 99)
    assert f.status == 'ok'
    fitted = fit_model('GARCH', returns(p), (0,0))
    expected = fitted.forecast(horizon=20).variance.iloc[-1].to_numpy()
    actual = np.array([f.values[h]['return_variance_percent2'] for h in range(1,21)])
    np.testing.assert_allclose(actual, expected)
    params = f.details['parameters']
    np.testing.assert_allclose(actual[1:], params['omega']+(params['alpha[1]']+params['beta[1]'])*actual[:-1])


@pytest.mark.parametrize('model', ['Naive', 'ARIMA', 'GARCH'])
def test_future_mutations_do_not_change_origin(prices, model):
    changed = prices.copy()
    changed.iloc[300:] *= 5
    assert forecast(model, prices.iloc[:300], 5, 10).values == forecast(model, changed.iloc[:300], 5, 10).values
