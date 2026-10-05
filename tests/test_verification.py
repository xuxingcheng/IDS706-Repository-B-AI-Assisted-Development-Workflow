"""The audit must reject corrupt artifacts even if scores are regenerated."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('verify_run', Path('scripts/verify_run.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def records():
    dates = pd.bdate_range('2024-01-01', periods=8)
    data = pd.DataFrame({'GLD': np.arange(100., 108.)}, index=dates)
    config = dict(target='GLD', train_end=str(dates[1].date()),
                  validation_end=str(dates[4].date()), horizons=[1, 5, 20])
    rows = []
    for o in range(4, 7):
        for model in audit.MODELS:
            rows.append(dict(model=model, phase='test', origin=str(dates[o].date()),
                             training_end=str(dates[o].date()), horizon=1,
                             target_date=str(dates[o+1].date()), actual_price=101.+o,
                             actual_return_percent=100*np.log((101.+o)/(100.+o)),
                             fit_status='ok', point_price=100.+o, lower95=90., lower80=95.,
                             upper80=110., upper95=115., return_variance_percent2=1.,
                             return_variance_decimal2=.0001))
    return pd.DataFrame(rows), data, config


def test_complete_records_pass(records):
    rows, data, config = records
    audit.verify_records(rows, data, config, 'test')


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'actual', 'return', 'variance', 'bounds'])
def test_corrupt_records_rejected(records, corruption):
    rows, data, config = records
    if corruption == 'missing': rows = rows.iloc[1:]
    if corruption == 'duplicate': rows = pd.concat([rows, rows.iloc[:1]])
    if corruption == 'actual': rows.loc[0, 'actual_price'] += 1
    if corruption == 'return': rows.loc[0, 'actual_return_percent'] += 1
    if corruption == 'variance': rows.loc[0, 'return_variance_percent2'] = np.inf
    if corruption == 'bounds': rows.loc[0, 'lower95'] = 120.
    with pytest.raises(AssertionError):
        audit.verify_records(rows, data, config, 'test')
