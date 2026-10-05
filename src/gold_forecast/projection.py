"""Optional calendar-dated 2026 projection, separate from the frozen backtest."""
import base64
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .models import forecast
from .metrics import MODELS, LEGACY_MODELS

CALENDAR_SOURCE = 'https://ir.theice.com/press/news-details/2024/NYSE-Group-Announces-2025-2026-and-2027-Holiday-and-Early-Closings-Calendar/default.aspx'
# Published full-day closures in the supported forecast window. Early closes remain sessions.
HOLIDAYS = ['2025-09-01', '2025-11-27', '2025-12-25',
            '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
            '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
            '2026-11-26', '2026-12-25']
NOTE = ('Historical as-of forecast from 2025-08-14; no later actual prices were supplied. '
        'Future dates assume the published NYSE/NYSE Arca holiday schedule; the CSV instrument and historical calendar remain unverified. '
        'Early-close days count as sessions. Exceptional closures are not modeled. '
        'This long-horizon extrapolation was not validated by the 1/5/20-session backtest. '
        'Gaussian fitted-parameter bands omit parameter uncertainty; GLD proxy units are unverified.')


def projection_dates(origin):
    origin = pd.Timestamp(origin)
    if origin != pd.Timestamp('2025-08-14'):
        raise ValueError('This calendar projection is explicitly scoped to the supplied 2025-08-14 cutoff')
    return pd.date_range(origin + pd.Timedelta(days=1), '2026-12-31',
                         freq=pd.offsets.CustomBusinessDay(holidays=HOLIDAYS))


def generate_projection(prices, spec, directory):
    """Refit frozen specifications once, without using future prices or retuning."""
    directory = Path(directory)
    target = directory / 'projection_2026'
    dates = projection_dates(prices.index[-1])
    config = spec['config']
    rows, fits = [], {}
    models = [m for m in spec.get('models', LEGACY_MODELS) if m != 'Naive']
    for model in models:
        seed = config['seed'] + 100000 + MODELS.index(model)
        result = forecast(model, prices, len(dates), seed, config['simulations'], spec['order'])
        fits[model] = dict(status=result.status, **result.details)
        if result.status != 'ok':
            raise ValueError(f'{model} projection failed: {result.details}')
        for h, date in enumerate(dates, start=1):
            rows.append(dict(model=model, phase='projection_2026', origin=str(prices.index[-1].date()),
                             target_date=str(date.date()), horizon=h, actual_price=np.nan,
                             fit_status=result.status, seed=seed, **result.values[h]))
    target.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(target / 'forecasts.csv', index=False)
    prices.loc['2024-01-01':].rename('observed_price').to_csv(target / 'observations.csv')
    metadata = dict(as_of=str(prices.index[-1].date()), forecast_end=str(dates[-1].date()),
                    steps=len(dates), sessions_in_2026=int((dates.year == 2026).sum()),
                    order=list(spec['order']), simulations=config['simulations'],
                    calendar_source=CALENDAR_SOURCE, calendar_verified_on='2026-10-04',
                    calendar_assumption='NYSE/NYSE Arca published schedule; not verified instrument provenance',
                    holidays=HOLIDAYS, note=NOTE, fits=fits)
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Plot persisted records so exported data and the figure agree.
    predicted = pd.read_csv(target / 'forecasts.csv', parse_dates=['target_date'])
    observed = pd.read_csv(target / 'observations.csv', index_col=0, parse_dates=True).observed_price
    fig, axes = plt.subplots(len(models), 1, figsize=(13, 4 * len(models)), sharex=True, sharey=True)
    origin = observed.index[-1]
    colors = {'ARIMA': '#1565c0', 'GARCH': '#c34b13', 'AR': '#238b45'}
    for ax, model in zip(axes, models):
        color = colors[model]
        f = predicted[predicted.model == model]
        x = pd.DatetimeIndex([origin, *f.target_date])
        def anchored(field):
            return np.r_[observed.iloc[-1], f[field].to_numpy()]
        ax.axvspan(pd.Timestamp('2026-01-01'), pd.Timestamp('2027-01-01'), color='#f1f5f9', zorder=0)
        ax.fill_between(x, anchored('lower95'), anchored('upper95'), color=color, alpha=.13, label='95% prediction interval')
        ax.fill_between(x, anchored('lower80'), anchored('upper80'), color=color, alpha=.23, label='80% prediction interval')
        ax.plot(observed.index, observed, color='#171717', lw=1.5, label='Observed 2024–2025')
        ax.plot(x, anchored('point_price'), color=color, lw=2.0, linestyle='--', label=f'{model} forecast median')
        ax.axvline(origin, color='#475569', linestyle=':', lw=1.2)
        ax.axvline(pd.Timestamp('2026-01-01'), color='#94a3b8', lw=.8)
        ax.set_title(f'{model} — observed trend and forecast', loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel('GLD proxy price (units unverified)')
        ax.grid(axis='y', alpha=.18)
        ax.legend(loc='upper left', ncol=2, fontsize=9)
        ax.annotate(f'Dec 2026 median: {f.point_price.iloc[-1]:.1f}',
                    xy=(dates[-1], f.point_price.iloc[-1]), xytext=(-10, 14),
                    textcoords='offset points', ha='right', fontsize=9, color=color)
    axes[0].text(pd.Timestamp('2025-09-01'), .97, 'Forecast starts after 14 Aug 2025',
                 transform=axes[0].get_xaxis_transform(), va='top', fontsize=9, color='#475569')
    axes[-1].set_xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2027-01-01'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.suptitle('Gold-price proxy: 2024–2025 history and projections through 2026', fontsize=16, fontweight='bold')
    fig.text(.06, .025, 'Observed data ends 14 Aug 2025. All later values are forecasts, not actual 2026 data.\n'
             'Assumed NYSE calendar • Long-horizon extrapolation is untested • Bands omit parameter uncertainty',
             fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .07, 1, .95])
    fig.savefig(target / 'forecast_2024_2026.png', dpi=160)
    fig.savefig(target / 'forecast_2024_2026.svg')
    plt.close(fig)
    return predicted


def extension_section(directory):
    path = Path(directory) / 'projection_2026' / 'forecast_2024_2026.png'
    if not path.exists():
        return ''
    encoded = base64.b64encode(path.read_bytes()).decode()
    return (f'<section id="projection-2026"><h2>2024–2025 trend and forecasts through 2026</h2>'
            f'<p>{NOTE}</p><p>Calendar reference: <a href="{CALENDAR_SOURCE}">NYSE published holiday schedule</a>.</p>'
            f'<img alt="Observed 2024–2025 GLD proxy prices and model projections through 2026" '
            f'src="data:image/png;base64,{encoded}"></section>')
