"""Training-only diagnostics and simulation precision audit."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from .data import returns
from .models import fit_model, forecast
from .backtest import save_json


def diagnostics(prices, directory, order=(0, 0)):
    r = returns(prices)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = dict(adf_pvalue=adfuller(r)[1], kpss_pvalue=kpss(r, regression='c', nlags='auto')[1],
                      arch_pvalue=het_arch(r, nlags=10)[1], acf=acf(r, nlags=20).tolist(),
                      pacf=pacf(r, nlags=20).tolist(), training_end=str(prices.index[-1].date()))
        residuals = {}
        for name in ('ARIMA', 'GARCH', 'AR'):
            try:
                fit = fit_model(name, r, order)
                residual = np.asarray(fit.std_resid if name == 'GARCH' else fit.resid)
                residuals[name] = residual
                result[name] = dict(ljung_box_pvalue=acorr_ljungbox(residual, lags=[10], model_df=1 if name == 'AR' else sum(order) if name == 'ARIMA' else 0, return_df=True).lb_pvalue.iloc[0],
                                    squared_ljung_box_pvalue=acorr_ljungbox(residual ** 2, lags=[10], return_df=True).lb_pvalue.iloc[0],
                                    arch_pvalue=het_arch(residual, nlags=10)[1])
            except ValueError as exc:
                result[name] = {'error': str(exc)}
        result['warnings'] = [str(w.message) for w in caught]
    save_json(directory / 'diagnostics.json', result)
    pd.DataFrame({'return_percent': r, **residuals}, index=prices.index[1:]).to_csv(directory / 'training_residuals.csv')


def stability(prices, config, directory):
    """Pre-test audit on initial-training origins; does not change frozen settings."""
    rows = []
    training = prices.loc[:config['train_end']]
    for offset in (0, 20):
        history = training.iloc[:len(training)-offset]
        for model in ('Naive', 'ARIMA', 'GARCH', 'AR'):
            a = forecast(model, history, 20, config['seed'], config['simulations'])
            b = forecast(model, history, 20, config['seed'] + 1, config['simulations'] * 2)
            for h in config['horizons']:
                row = dict(model=model, origin=str(history.index[-1].date()), horizon=h,
                           base_seed=config['seed'], audit_seed=config['seed']+1,
                           base_paths=config['simulations'], audit_paths=config['simulations']*2,
                           base_status=a.status, audit_status=b.status)
                if a.status == b.status == 'ok':
                    row.update({f'{key}_relative_change': abs(b.values[h][key]-a.values[h][key])/float(history.iloc[-1])
                                for key in ('point_price', 'lower80', 'upper80', 'lower95', 'upper95')})
                rows.append(row)
    pd.DataFrame(rows).to_csv(directory / 'simulation_stability.csv', index=False)
