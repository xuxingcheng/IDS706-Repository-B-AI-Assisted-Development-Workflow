"""Common-pair scoring and paired chronological moving-block inference."""
from itertools import combinations
import numpy as np
import pandas as pd

VARIANCE_FLOOR = 1e-12  # percentage-return squared units
LEGACY_MODELS = ['Naive', 'ARIMA', 'GARCH']
# Append models to preserve the original models' recorded seed offsets.
MODELS = [*LEGACY_MODELS, 'AR']


def model_set(frame):
    """Old artifacts remain readable; AR rows opt into four-model pairing."""
    return MODELS if frame.model.eq('AR').any() else LEGACY_MODELS
SCORE_FIELDS = ['mae', 'rmse', 'error_min', 'error_max', 'error_median', 'error_p05', 'error_p95', 'qlike'] + [f'{name}{level}' for level in (80, 95) for name in ('coverage', 'width', 'interval_score')]


def interval_score(actual, lower, upper, alpha):
    return upper - lower + 2 / alpha * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))


def qlike(realized_return, variance):
    variance = np.asarray(variance)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('Variance must be positive and finite')
    v = np.maximum(variance, VARIANCE_FLOOR)
    return np.log(v) + np.asarray(realized_return) ** 2 / v


def common_pairs(frame):
    valid = frame[frame.fit_status.eq('ok') & frame.actual_price.notna()].copy()
    counts = valid.groupby(['origin', 'horizon']).model.nunique()
    keys = counts[counts == len(model_set(frame))].index
    return valid.set_index(['origin', 'horizon']).loc[lambda x: x.index.isin(keys)].reset_index()


def summarize(frame):
    common = common_pairs(frame)
    rows = []
    for (model, horizon), attempted in frame.groupby(['model', 'horizon']):
        f = common[(common.model == model) & (common.horizon == horizon)]
        row = dict(model=model, horizon=horizon, attempted=len(attempted),
                   failures=int(attempted.fit_status.ne('ok').sum()), n=len(f),
                   valid_coverage=float(attempted.fit_status.eq('ok').mean()))
        row.update({key: np.nan for key in SCORE_FIELDS})
        if len(f):
            error = f.actual_price - f.point_price
            row.update(mae=float(abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())),
                       error_min=error.min(), error_max=error.max(), error_median=error.median(),
                       error_p05=error.quantile(.05), error_p95=error.quantile(.95))
            for level in (80, 95):
                lower, upper = f[f'lower{level}'], f[f'upper{level}']
                row.update({f'coverage{level}': ((f.actual_price >= lower) & (f.actual_price <= upper)).mean(),
                            f'width{level}': (upper - lower).mean(),
                            f'interval_score{level}': interval_score(f.actual_price, lower, upper, 1 - level / 100).mean()})
            if horizon == 1:
                row['qlike'] = qlike(f.actual_return_percent, f.return_variance_percent2).mean()
        rows.append(row)
    scores = pd.DataFrame(rows)
    if 'mae' in scores:
        baseline = scores[scores.model == 'Naive'].set_index('horizon').mae
        scores['mae_improvement_vs_naive_percent'] = [100 * (1 - r.mae / baseline[r.horizon]) if baseline[r.horizon] else np.nan for r in scores.itertuples()]
    return scores


def moving_block_ci(difference, block, seed, samples=2000):
    d = np.asarray(difference, dtype=float)
    if len(d) == 0 or not np.isfinite(d).all():
        raise ValueError('Finite, paired losses required')
    block = min(block, len(d))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(d) - block + 1, size=(samples, int(np.ceil(len(d) / block))))
    indexes = (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :len(d)]
    return np.quantile(d[indexes].mean(axis=1), [.025, .975])


def paired_analysis(frame, config):
    common = common_pairs(frame)
    rows = []
    for horizon, f in common.groupby('horizon'):
        f = f.copy()
        f['absolute_loss'] = abs(f.actual_price - f.point_price)
        for level in (80, 95):
            f[f'interval{level}'] = interval_score(f.actual_price, f[f'lower{level}'], f[f'upper{level}'], 1-level/100)
        for loss in ('absolute_loss', 'interval80', 'interval95'):
            pivot = f.pivot(index='origin', columns='model', values=loss).sort_index()
            for a, b in combinations(model_set(frame), 2):
                d = pivot[a] - pivot[b]
                for block in config['bootstrap_blocks']:
                    seed = config['seed'] + int(horizon) * 100 + block
                    low, high = moving_block_ci(d, block, seed, config['bootstrap_samples'])
                    rows.append(dict(horizon=horizon, loss=loss, model_a=a, model_b=b,
                                     mean_difference=d.mean(), lower95=low, upper95=high,
                                     block=block, effective_block=min(block, len(d)),
                                     inference_status='ok' if len(d) >= 2 * block else 'limited_sample',
                                     seed=seed, n=len(d)))
    return pd.DataFrame(rows, columns=['horizon', 'loss', 'model_a', 'model_b', 'mean_difference', 'lower95', 'upper95', 'block', 'effective_block', 'inference_status', 'seed', 'n'])
