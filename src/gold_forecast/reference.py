"""Add a fixed AR reference to a separate copy of a completed legacy run."""
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from .backtest import rolling, final_forecasts, save_json
from .data import load_data
from .metrics import MODELS, LEGACY_MODELS


def add_ar_reference(source, destination):
    source, destination = Path(source), Path(destination)
    spec = json.loads((source / 'frozen.json').read_text())
    source_manifest = json.loads((source / 'manifest.json').read_text())
    if spec.get('models', LEGACY_MODELS) != LEGACY_MODELS:
        raise ValueError('Reference extension requires an original three-model run')
    config = spec['config']
    data, quality = load_data(config['input'], config)
    if quality['sha256'] != source_manifest['input_sha256'] or config != source_manifest['config']:
        raise ValueError('Input/configuration differs from source run')
    original = {}
    for phase in ('validation', 'test', 'final'):
        frame = pd.read_csv(source / f'{phase}_forecasts.csv')
        if set(frame.model) != set(LEGACY_MODELS):
            raise ValueError(f'Unexpected source model set in {phase}')
        original[phase] = frame
    # Reuse the frozen original forecasts; no test-driven reselection or refitting.
    from .cli import initialize
    from .report import generate
    from .diagnostics import diagnostics, stability
    started = time.perf_counter()
    initialize(config, destination, quality)
    save_json(destination / 'source_manifest.json', source_manifest)
    for pattern in ('validation_arima_*.csv', 'tuning_scores.csv', 'training_residuals.csv',
                    'simulation_stability.csv', 'diagnostics.json'):
        for path in source.glob(pattern):
            if path.suffix == '.csv':
                frame = pd.read_csv(path)
                if 'run_id' in frame:
                    frame['run_id'] = destination.name
                frame.to_csv(destination / path.name, index=False)
            else:
                shutil.copy2(path, destination / path.name)
    spec.update(models=MODELS, ar_order=1, reference_extension=dict(
        source_run=str(source.resolve()),
        note='Fixed AR(1) added after original evaluation; AR comparisons are exploratory.'))
    save_json(destination / 'frozen.json', spec)
    prices = data[config['target']]
    diagnostics(prices.loc[:config['train_end']], destination, spec['order'])
    stability(prices, config, destination)
    for phase in ('validation', 'test', 'final'):
        if phase == 'final':
            added = final_forecasts(prices, spec, destination, models=['AR'])
        else:
            history = prices.loc[:config['validation_end']] if phase == 'validation' else prices
            boundary = config['train_end' if phase == 'validation' else 'validation_end']
            start = int((history.index <= boundary).sum())
            added = rolling(history, start, len(history), config, phase, destination.name, spec['order'], ['AR'])
        old = original[phase].copy()
        old['run_id'] = destination.name
        pd.concat([old, added], ignore_index=True).to_csv(destination / f'{phase}_forecasts.csv', index=False)
    if (source / 'projection_2026').exists():
        from .projection import generate_projection
        generate_projection(prices, spec, destination)
    generate(destination)
    manifest = json.loads((destination / 'manifest.json').read_text())
    manifest.update(selected_parameters=spec['order'], models=MODELS, ar_order=1,
                    reference_extension=spec['reference_extension'],
                    elapsed_seconds=time.perf_counter() - started)
    save_json(destination / 'manifest.json', manifest)
    return destination
