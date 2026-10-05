from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
import pytest
from gold_forecast.data import load_data, returns, price_paths, origins

CONFIG = tomllib.loads(Path('configs/default.toml').read_text())


def test_real_data_contract():
    data, quality = load_data(CONFIG['input'], CONFIG)
    assert quality['split_counts'] == [1760, 501, 405]
    assert quality['end'] == '2025-08-14'
    assert len(data) == 2666
    assert quality['weekend_observations'] == 0


@pytest.mark.parametrize('mutation', ['date', 'duplicate', 'missing', 'zero', 'infinite', 'column', 'reverse'])
def test_invalid_data(tmp_path, mutation):
    f = pd.read_csv(CONFIG['input'])
    if mutation == 'date': f.loc[2, 'Date'] = 'not a date'
    if mutation == 'duplicate': f.loc[2, 'Date'] = f.loc[1, 'Date']
    if mutation == 'missing': f.loc[2, 'GLD'] = np.nan
    if mutation == 'zero': f.loc[2, 'GLD'] = 0
    if mutation == 'infinite': f.loc[2, 'GLD'] = np.inf
    if mutation == 'column': f = f.drop(columns='USO')
    if mutation == 'reverse': f = f.iloc[::-1]
    path = tmp_path / 'bad.csv'
    f.to_csv(path, index=False)
    with pytest.raises(ValueError): load_data(path, CONFIG)


def test_transform_and_alignment():
    p = [100, 110, 99]
    r = returns(p)
    np.testing.assert_allclose(price_paths(100, r[None, :]), [[110, 99]])
    assert r[0] == pytest.approx(100*np.log(1.1))
    pairs = list(origins(pd.date_range('2020-01-01', periods=30), 10, 25, [1,5,20]))
    assert pairs[0] == (9, [1,5])
    assert all(10 <= origin+h < 25 for origin, hs in pairs for h in hs)
    assert len(pairs) == 15
