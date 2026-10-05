"""Expanding histories and immutable validation-selected specifications."""
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import numpy as np
import pandas as pd
from .data import origins
from .models import forecast, FORECAST_FIELDS
from .metrics import MODELS, LEGACY_MODELS


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def rolling(prices, start, end, config, phase, run_id, order, models=MODELS):
    rows = []
    for origin, horizons in origins(prices.index, start, end, config['horizons']):
        history = prices.iloc[:origin + 1].copy()
        for model in models:
            seed = int(config['seed'] + origin * 31 + MODELS.index(model))
            result = forecast(model, history, max(horizons), seed, config['simulations'], order)
            for h in horizons:
                target = origin + h
                rows.append(dict(run_id=run_id, phase=phase, model=model,
                    origin=str(history.index[-1].date()), training_end=str(history.index[-1].date()),
                    horizon=h, target_date=str(prices.index[target].date()), actual_price=prices.iloc[target],
                    actual_return_percent=100 * np.log(prices.iloc[target] / prices.iloc[origin]) if h == 1 else np.nan,
                    fit_status=result.status, seed=seed, fit_details=json.dumps(result.details),
                    **result.values.get(h, dict.fromkeys(FORECAST_FIELDS, np.nan))))
        completed = origin - start + 2
        if completed % 50 == 0:
            print(f'{phase} {models} order={tuple(order)}: {completed}/{end-start} origins', flush=True)
    return pd.DataFrame(rows)


def _candidate(args):
    prices, start, config, run_id, order, path = args
    # The worker receives validation-and-earlier observations only, never test prices.
    frame = rolling(prices, start, len(prices), config, 'validation', run_id, order, ['ARIMA'])
    frame.to_csv(path, index=False)
    one = frame[frame.horizon == 1]
    ok = one.fit_status.eq('ok')
    mae = float(abs(one.loc[ok, 'actual_price'] - one.loc[ok, 'point_price']).mean()) if ok.any() else None
    print(f'Completed ARIMA {order}: validation MAE={mae}, failures={int((~ok).sum())}/{len(one)}', flush=True)
    return dict(p=order[0], q=order[1], mae=mae, attempted=len(one), failures=int((~ok).sum()), file=path.name)


def tune(prices, config, directory, workers=1):
    frozen = directory / 'frozen.json'
    if frozen.exists():
        raise ValueError('This run is already frozen; create a new run to retune')
    validation = prices.loc[:config['validation_end']]
    start = int((validation.index <= config['train_end']).sum())
    jobs = [(validation, start, config, directory.name, (p, q), directory / f'validation_arima_{p}_{q}.csv')
            for p, q in product(config['orders'], repeat=2)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_candidate, jobs))
    else:
        results = [_candidate(job) for job in jobs]
    table = pd.DataFrame(results)
    table.to_csv(directory / 'tuning_scores.csv', index=False)
    eligible = table[table.failures == 0].sort_values(['mae', 'p', 'q'])
    if eligible.empty:
        raise ValueError('No ARIMA candidate has complete valid validation coverage; inspect tuning failures')
    best = eligible.iloc[0]
    spec = dict(order=[int(best.p), int(best.q)], models=MODELS, ar_order=1, criterion='validation one-session price MAE',
                config=config, selected_file=best.file, validation_end=str(validation.index[-1].date()))
    save_json(frozen, spec)
    selected = pd.read_csv(directory / best.file)
    others = rolling(validation, start, len(validation), config, 'validation', directory.name, spec['order'], ['Naive', 'GARCH', 'AR'])
    pd.concat([selected, others], ignore_index=True).to_csv(directory / 'validation_forecasts.csv', index=False)
    return spec


def evaluate(prices, spec, directory):
    path = directory / 'test_forecasts.csv'
    if path.exists():
        raise ValueError('Test artifacts already exist; refusing to overwrite evaluation')
    config = spec['config']
    start = int((prices.index <= config['validation_end']).sum())
    frame = rolling(prices, start, len(prices), config, 'test', directory.name, spec['order'], spec.get('models', LEGACY_MODELS))
    frame.to_csv(path, index=False)
    return frame


def final_forecasts(prices, spec, directory, models=None):
    rows = []
    config = spec['config']
    for model in (models if models is not None else spec.get('models', LEGACY_MODELS)):
        seed = config['seed'] + 100000 + MODELS.index(model)
        result = forecast(model, prices, 20, seed, config['simulations'], spec['order'])
        for h in range(1, 21):
            rows.append(dict(run_id=directory.name, phase='final', model=model, origin=str(prices.index[-1].date()),
                training_end=str(prices.index[-1].date()), horizon=h, target_date=None, actual_price=None,
                actual_return_percent=None, fit_status=result.status, seed=seed,
                fit_details=json.dumps(result.details), **result.values.get(h, dict.fromkeys(FORECAST_FIELDS, np.nan))))
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / 'final_forecasts.csv', index=False)
    return frame
