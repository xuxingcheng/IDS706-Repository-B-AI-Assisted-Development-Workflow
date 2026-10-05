"""CLI stages guard input/configuration identity and preserve test records."""
import argparse
import hashlib
import time
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tomllib
from .data import load_data
from .backtest import save_json, tune, evaluate, final_forecasts
from .diagnostics import diagnostics, stability


def initialize(config, directory, quality):
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / 'quality_report.json', quality)
    save_json(directory / 'manifest.json', dict(run_id=directory.name, input_sha256=quality['sha256'],
        config=config, lockfile_sha256=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest() if Path('uv.lock').exists() else None, timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        packages={p: version(p) for p in ['gold-forecast', 'numpy', 'pandas', 'scipy', 'statsmodels', 'arch', 'matplotlib']},
        seed_rule='base + origin_offset * 31 + model_index; final base + 100000 + model_index',
        variance_units='percentage log-return squared; decimal variance divides by 10000',
        numerical_tolerance='same lockfile/platform: rtol=1e-8, atol=1e-10',
        retry_policy='ARIMA maxiter 300, then same specification/start parameters maxiter 1000; GARCH no retry'))


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'tune', 'evaluate', 'forecast', 'report', 'run', 'project-2026'])
    parser.add_argument('--config', default='configs/default.toml')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--workers', type=int, default=1, help='Independent ARIMA candidate processes')
    args = parser.parse_args(argv)
    config = tomllib.loads(Path(args.config).read_text())
    if config['simulations'] < 10000 or config['horizons'] != [1, 5, 20]:
        raise ValueError('Production contract requires >=10000 paths and horizons [1,5,20]')
    frame, quality = load_data(config['input'], config)
    prices = frame[config['target']]
    if args.command == 'validate':
        print(json.dumps(quality, indent=2))
        return
    if args.command not in ('run', 'tune') and not args.run_id:
        parser.error('--run-id is required for existing-run stages')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')
    directory = Path(config['output']) / run_id
    if args.command in ('run', 'tune'):
        initialize(config, directory, quality)
        print(f'Created {directory}; running training diagnostics and precision audit', flush=True)
        diagnostics(prices.loc[:config['train_end']], directory)
        stability(prices, config, directory)
        print('Tuning ARIMA on validation only', flush=True)
        spec = tune(prices, config, directory, args.workers)
        diagnostics(prices.loc[:config['train_end']], directory, spec['order'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['selected_parameters'] = spec['order']
        manifest['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(directory / 'manifest.json', manifest)
        print(f'Frozen ARIMA order: {spec["order"]}', flush=True)
    else:
        manifest = json.loads((directory / 'manifest.json').read_text())
        spec = json.loads((directory / 'frozen.json').read_text())
        if quality['sha256'] != manifest['input_sha256'] or config != spec['config']:
            raise ValueError('Input/configuration differs from frozen run')
    if args.command in ('run', 'evaluate'):
        print('Evaluating frozen specifications on test observations', flush=True)
        evaluate(prices, spec, directory)
    if args.command in ('run', 'forecast'):
        if not (directory / 'test_forecasts.csv').exists():
            raise ValueError('Evaluate before final refitting')
        final_forecasts(prices, spec, directory)
    if args.command == 'project-2026':
        if not (directory / 'test_forecasts.csv').exists():
            raise ValueError('Complete evaluation before projecting through 2026')
        from .projection import generate_projection
        generate_projection(prices, spec, directory)
    if args.command in ('run', 'report', 'project-2026'):
        from .report import generate
        generate(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    manifest.setdefault('executions', []).append(dict(command=args.command, elapsed_seconds=time.perf_counter()-started, completed_at_utc=datetime.now(timezone.utc).isoformat()))
    save_json(directory / 'manifest.json', manifest)
    print(directory, flush=True)


if __name__ == '__main__':
    main()
