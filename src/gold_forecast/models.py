"""Conditional path forecasts in percentage log-return units."""
from dataclasses import dataclass
import warnings
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from .data import returns, price_paths


FORECAST_FIELDS = ['point_price', 'lower80', 'upper80', 'lower95', 'upper95',
                   'return_variance_percent2', 'return_variance_decimal2']


@dataclass
class Forecast:
    values: dict
    status: str
    details: dict


def fit_model(name, r, order):
    if name in ('AR', 'ARIMA'):
        # Fixed AR(1) reference; never reuse the validation-selected ARIMA order.
        order = (1, 0) if name == 'AR' else order
        # Fixed training-only retry policy: increase optimizer iterations, same specification.
        fit = ARIMA(r, order=(order[0], 0, order[1]), trend='c').fit(method_kwargs={'maxiter': 300})
        if not fit.mle_retvals.get('converged', False):
            fit = ARIMA(r, order=(order[0], 0, order[1]), trend='c').fit(
                start_params=fit.params, method_kwargs={'maxiter': 1000})
        if not fit.mle_retvals.get('converged', False):
            raise ValueError('ARIMA optimizer did not converge after fixed retry')
        return fit
    if name != 'GARCH':
        raise ValueError(f'Unknown fitted model: {name}')
    fit = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal', rescale=False).fit(disp='off')
    p = fit.params
    if fit.convergence_flag or p['omega'] <= 0 or min(p['alpha[1]'], p['beta[1]']) < 0 or p['alpha[1]'] + p['beta[1]'] >= 1:
        raise ValueError('Invalid/nonstationary GARCH fit or optimizer failure')
    return fit


def arima_distribution(fit, horizon):
    """Exact Gaussian joint forecast covariance, including filtered-state uncertainty."""
    prediction = fit.get_forecast(horizon)
    variance = np.asarray(prediction.var_pred_mean)
    transition = fit.model.ssm['transition']
    design = fit.model.ssm['design']
    selection = fit.model.ssm['selection']
    noise = selection @ fit.model.ssm['state_cov'] @ selection.T
    state_cov = fit.filter_results.predicted_state_cov[:, :, -1].copy()
    covariance = np.diag(variance)
    for i in range(horizon):
        propagated = state_cov.copy()
        for j in range(i + 1, horizon):
            propagated = transition @ propagated
            covariance[j, i] = covariance[i, j] = (design @ propagated @ design.T).item()
        state_cov = transition @ state_cov @ transition.T + noise
    return np.asarray(prediction.predicted_mean), covariance


def forecast(name, prices, horizon, seed, simulations=10000, order=(0, 0)):
    if name not in ('Naive', 'ARIMA', 'GARCH', 'AR') or horizon < 1 or simulations < 10000:
        raise ValueError('Known model, positive horizon and at least 10000 paths required')
    details = {'seed': int(seed), 'simulations': simulations}
    if name == 'AR':
        details['specification'] = 'Fixed AR(1) with constant on percentage log returns; Gaussian innovations'
        details['order'] = [1, 0, 0]
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = returns(prices)
            rng = np.random.default_rng(seed)
            # Antithetic paths reduce Monte Carlo median noise while preserving Gaussian marginals.
            z = rng.normal(size=((simulations + 1) // 2, horizon))
            z = np.concatenate([z, -z])[:simulations]
            if name == 'Naive':
                variance = np.full(horizon, np.var(r, ddof=1))
                paths = z * np.sqrt(variance)
                details['parameters'] = {'variance_percent2': float(variance[0])}
            else:
                fit = fit_model(name, r, order)
                details.update(parameters=dict(zip(fit.model.param_names, map(float, fit.params))) if name in ('AR', 'ARIMA') else fit.params.to_dict(),
                               aic=float(fit.aic), bic=float(fit.bic))
                if name in ('AR', 'ARIMA'):
                    mean, covariance = arima_distribution(fit, horizon)
                    paths = mean + z @ np.linalg.cholesky(covariance).T
                    variance = np.diag(covariance)
                else:
                    p = fit.params
                    omega, alpha, beta = p['omega'], p['alpha[1]'], p['beta[1]']
                    v = np.full(simulations, omega + alpha * fit.resid[-1] ** 2 + beta * fit.conditional_volatility[-1] ** 2)
                    paths = np.empty_like(z)
                    variance = np.asarray(fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1])
                    for h in range(horizon):
                        shock = np.sqrt(v) * z[:, h]
                        paths[:, h] = p['mu'] + shock
                        v = omega + alpha * shock ** 2 + beta * v
                    details['persistence'] = float(alpha + beta)
            prices_simulated = price_paths(float(prices.iloc[-1]), paths)
            quantiles = np.quantile(prices_simulated, [.025, .1, .5, .9, .975], axis=0)
            if not np.isfinite(quantiles).all() or not np.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError('Nonfinite forecast or nonpositive variance')
            values = {}
            for h in range(horizon):
                lo95, lo80, point, hi80, hi95 = quantiles[:, h]
                values[h + 1] = dict(point_price=float(prices.iloc[-1]) if name == 'Naive' else point,
                    lower80=lo80, upper80=hi80, lower95=lo95, upper95=hi95,
                    return_variance_percent2=variance[h], return_variance_decimal2=variance[h] / 10000)
        status = 'ok'
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        values, status = {}, 'failed'
        details['error'] = str(exc)
    details['warnings'] = [str(w.message) for w in caught]
    return Forecast(values, status, details)
