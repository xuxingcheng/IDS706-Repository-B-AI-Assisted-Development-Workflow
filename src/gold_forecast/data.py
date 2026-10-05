"""Strict input validation; observations are never filled or calendar-expanded."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def load_data(path, config):
    raw = Path(path).read_bytes()
    frame = pd.read_csv(path)
    required = {'Date', 'SPX', 'GLD', 'USO', 'SLV', 'EUR/USD'}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {required - set(frame)}')
    frame['Date'] = pd.to_datetime(frame.Date, format='%Y-%m-%d', errors='raise')
    if frame.Date.isna().any() or frame.Date.duplicated().any() or not frame.Date.is_monotonic_increasing:
        raise ValueError('Dates must be nonmissing, unique, and strictly ascending')
    values = frame[list(required - {'Date'})].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values).all().all() or (values <= 0).any().any():
        raise ValueError('All asset values must be finite and positive')
    if config['target'] != 'GLD':
        raise ValueError('This experiment supports only the provisional GLD target')
    frame = frame.set_index('Date')
    counts = [int((frame.index <= config['train_end']).sum()),
              int(((frame.index > config['train_end']) & (frame.index <= config['validation_end'])).sum()),
              int((frame.index > config['validation_end']).sum())]
    if min(counts) < 2:
        raise ValueError('Each chronological segment needs at least two observations')
    if 'expected_counts' in config and counts != config['expected_counts']:
        raise ValueError(f'Unexpected split counts: {counts}')
    gaps = frame.index.to_series().diff().dt.days
    report = dict(sha256=hashlib.sha256(raw).hexdigest(), observations=len(frame),
                  start=str(frame.index[0].date()), end=str(frame.index[-1].date()),
                  split_counts=counts, weekend_observations=int((frame.index.dayofweek >= 5).sum()),
                  gaps_over_four_days={str(k.date()): int(v) for k, v in gaps[gaps > 4].items()},
                  calendar='Unverified: use observed sessions only; no filling or missing-session claims',
                  target='GLD gold-price proxy; instrument, units, adjustment and licensing unverified')
    return frame, report


def returns(prices):
    return 100 * np.diff(np.log(np.asarray(prices, dtype=float)))


def price_paths(last_price, return_paths):
    return last_price * np.exp(np.cumsum(return_paths, axis=1) / 100)


def origins(index, start, end, horizons):
    """Yield origin offsets and eligible horizons; targets remain within the segment."""
    for origin in range(start - 1, end - 1):
        eligible = [h for h in horizons if start <= origin + h < end]
        if eligible:
            yield origin, eligible
