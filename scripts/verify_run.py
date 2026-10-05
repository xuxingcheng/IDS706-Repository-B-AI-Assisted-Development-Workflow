"""Audit a completed run and optionally compare a separately executed repeat run.

Run from the repository root: uv run python scripts/verify_run.py outputs/RUN [--compare outputs/REPEAT]
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
from gold_forecast.data import load_data, origins
from gold_forecast.metrics import summarize, MODELS, LEGACY_MODELS, common_pairs


def verify_records(rows, data, config, phase, models=MODELS):
    """Require every scheduled pair, including explicitly failed forecasts."""
    if phase == 'final':
        expected = {(model, str(data.index[-1].date()), h)
                    for model in models for h in range(1, 21)}
    else:
        start = int((data.index <= config['train_end' if phase == 'validation' else 'validation_end']).sum())
        end = int((data.index <= config['validation_end']).sum()) if phase == 'validation' else len(data)
        expected = {(model, str(data.index[o].date()), h)
                    for o, hs in origins(data.index, start, end, config['horizons'])
                    for h in hs for model in models}
    keys = ['model', 'origin', 'horizon']
    assert not rows.duplicated(keys).any(), 'Duplicate forecast records'
    assert set(rows[keys].itertuples(index=False, name=None)) == expected, 'Missing or unexpected forecast records'
    assert rows.phase.eq(phase).all()
    assert rows.training_end.eq(rows.origin).all()
    assert rows.fit_status.isin(['ok', 'failed']).all()
    valid = rows[rows.fit_status.eq('ok')]
    fields = ['lower95', 'lower80', 'point_price', 'upper80', 'upper95']
    assert np.isfinite(valid[fields]).all().all()
    assert (valid[fields] > 0).all().all()
    assert (np.diff(valid[fields].to_numpy(), axis=1) >= 0).all()
    assert np.isfinite(valid.return_variance_percent2).all()
    assert (valid.return_variance_percent2 > 0).all()
    np.testing.assert_allclose(valid.return_variance_percent2 / 10000, valid.return_variance_decimal2)
    if phase != 'final':
        actual = data.loc[pd.to_datetime(rows.target_date), config['target']].to_numpy()
        np.testing.assert_allclose(rows.actual_price, actual, rtol=1e-12, atol=1e-12)
        one = rows[rows.horizon.eq(1)]
        origin_prices = data.loc[pd.to_datetime(one.origin), config['target']].to_numpy()
        np.testing.assert_allclose(one.actual_return_percent, 100 * np.log(one.actual_price / origin_prices), rtol=1e-10, atol=1e-10)


def verify(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    spec = json.loads((directory / 'frozen.json').read_text())
    models = spec.get('models', LEGACY_MODELS)
    data, quality = load_data(spec['config']['input'], spec['config'])
    assert quality['sha256'] == manifest['input_sha256']
    expected = {'validation': spec['config']['validation_end'], 'test': quality['end']}
    for phase, end in expected.items():
        rows = pd.read_csv(directory / f'{phase}_forecasts.csv')
        verify_records(rows, data, spec['config'], phase, models)
        assert rows.phase.eq(phase).all()
        assert rows.training_end.eq(rows.origin).all()
        assert (rows.origin < rows.target_date).all()
        assert (rows.target_date <= end).all()
        start = spec['config']['train_end'] if phase == 'validation' else spec['config']['validation_end']
        assert (rows.target_date > start).all()
        for row in rows.itertuples():
            assert data.index.get_loc(row.target_date) - data.index.get_loc(row.origin) == row.horizon
        valid = rows[rows.fit_status.eq('ok')]
        fields = ['lower95','lower80','point_price','upper80','upper95']
        assert np.isfinite(valid[fields]).all().all()
        assert (np.diff(valid[fields].to_numpy(), axis=1) >= 0).all()
        assert (valid.return_variance_percent2 > 0).all()
        np.testing.assert_allclose(valid.return_variance_percent2/10000, valid.return_variance_decimal2)
        for row in valid[valid.model.eq('Naive')].itertuples():
            assert np.isclose(row.point_price, data.loc[row.origin, 'GLD'], rtol=1e-12, atol=1e-12)
    test = pd.read_csv(directory / 'test_forecasts.csv')
    scores = pd.read_csv(directory / 'scores.csv')
    pd.testing.assert_frame_equal(scores, summarize(test), check_dtype=False, rtol=1e-8, atol=1e-10)
    final = pd.read_csv(directory / 'final_forecasts.csv')
    verify_records(final, data, spec['config'], 'final', models)
    assert len(final) == 20 * len(models) and final.horizon.nunique() == 20
    assert final.actual_price.isna().all() and final.target_date.isna().all()
    assert final.origin.eq(quality['end']).all()
    plotted = pd.read_csv(directory / 'comparison_forecasts.csv')
    scored = plotted[plotted.phase.eq('test')]
    expected_plotted = pd.concat([common_pairs(test), final], ignore_index=True)
    pd.testing.assert_frame_equal(plotted.drop(columns='signed_error'), expected_plotted,
                                  check_dtype=False, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scored.signed_error, scored.actual_price-scored.point_price)
    pd.testing.assert_frame_equal(pd.read_csv(directory / 'comparison_metrics.csv'), scores)
    html = (directory / 'comparison.html').read_text()
    sources = re.findall(r'<img[^>]+src="([^"]+)"', html)
    projection = directory / 'projection_2026' / 'forecast_2024_2026.png'
    assert len(sources) == 9 + int(projection.exists())
    png_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory / 'comparison_plots').glob('*.png')}
    if projection.exists():
        png_hashes.add(hashlib.sha256(projection.read_bytes()).hexdigest())
    for source in sources:
        assert source.startswith('data:image/png;base64,')
        raw = base64.b64decode(source.split(',', 1)[1])
        assert hashlib.sha256(raw).hexdigest() in png_hashes
        Image.open(io.BytesIO(raw)).verify()
    return dict(run_id=directory.name, order=spec['order'], test_rows=len(test), final_rows=len(final),
                failed_test_rows=int(test.fit_status.ne('ok').sum()), embedded_plots=len(sources),
                input_sha256=quality['sha256'])


def compare(first, second):
    a = json.loads((first / 'frozen.json').read_text())
    b = json.loads((second / 'frozen.json').read_text())
    assert a == b
    files = ['tuning_scores.csv', 'validation_forecasts.csv', 'test_forecasts.csv', 'final_forecasts.csv',
             'scores.csv', 'paired_bootstrap.csv', 'comparison_metrics.csv', 'comparison_forecasts.csv']
    files += [p.name for p in sorted(first.glob('validation_arima_*.csv'))]
    for filename in files:
        left = pd.read_csv(first / filename).drop(columns=['run_id'], errors='ignore')
        right = pd.read_csv(second / filename).drop(columns=['run_id'], errors='ignore')
        pd.testing.assert_frame_equal(left, right, rtol=1e-8, atol=1e-10)
    return dict(compared_run=second.name, files_compared=len(files), rtol=1e-8, atol=1e-10, matched=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    if args.compare:
        verify(args.compare)
        result['reproducibility'] = compare(args.run, args.compare)
    (args.run / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
