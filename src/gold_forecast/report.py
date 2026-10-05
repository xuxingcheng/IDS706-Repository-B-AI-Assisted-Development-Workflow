"""Standalone reports generated exclusively from persisted forecast records."""
import base64
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .projection import extension_section
from .metrics import MODELS, common_pairs, summarize, paired_analysis

COLORS = {'Naive': '#6b7280', 'ARIMA': '#1565c0', 'GARCH': '#c34b13'}


def interpretation(scores, paired):
    one = scores[scores.horizon == 1].dropna(subset=['mae']).sort_values('mae')
    if one.empty:
        return 'No valid common forecast pairs; no winner can be assessed.'
    leader = one.iloc[0].model
    model_leader = one[one.model != 'Naive'].iloc[0].model
    text = [f'Lowest one-session price MAE: {leader}. Among ARIMA and GARCH: {model_leader}.']
    primary = paired[(paired.horizon == 1) & (paired.loss == 'absolute_loss') & (paired.block == 20)]
    for row in primary.itertuples():
        evidence = ('inconclusive (too few origins for this block length)' if row.n < 2 * row.block
                    else 'inconclusive (interval includes zero)' if row.lower95 <= 0 <= row.upper95
                    else f'favors {row.model_a if row.upper95 < 0 else row.model_b}')
        text.append(f'{row.model_a} minus {row.model_b} MAE difference: {row.mean_difference:.4f}, paired 95% CI [{row.lower95:.4f}, {row.upper95:.4f}]; evidence {evidence}.')
    if scores.failures.sum() or (scores.n < scores.attempted).any():
        text.append('Incomplete forecast coverage: rankings are conditional on common valid pairs; no unqualified winner.')
    for h, subset in scores.groupby('horizon'):
        for level in (80, 95):
            ranked = subset.dropna(subset=[f'interval_score{level}']).sort_values(f'interval_score{level}')
            if ranked.empty:
                continue
            best = ranked.iloc[0]
            text.append(f'At {h} sessions, lowest {level}% interval score: {best.model}, empirical coverage {best[f"coverage{level}"]:.1%}.')
    return ' '.join(text)


def generate(directory):
    config = json.loads((directory / 'frozen.json').read_text())['config']
    test = pd.read_csv(directory / 'test_forecasts.csv')
    final = pd.read_csv(directory / 'final_forecasts.csv') if (directory / 'final_forecasts.csv').exists() else pd.DataFrame()
    scores = summarize(test)
    paired = paired_analysis(test, config)
    scores.to_csv(directory / 'scores.csv', index=False)
    scores.to_csv(directory / 'comparison_metrics.csv', index=False)
    paired.to_csv(directory / 'paired_bootstrap.csv', index=False)
    common = common_pairs(test)
    plotted = pd.concat([common, final], ignore_index=True)
    plotted['signed_error'] = plotted.actual_price - plotted.point_price
    plotted.to_csv(directory / 'comparison_forecasts.csv', index=False)
    yearly = []
    for year, frame in test.groupby(test.target_date.str[:4]):
        summary = summarize(frame)
        summary['year'] = year
        yearly.append(summary)
    pd.concat(yearly).to_csv(directory / 'scores_by_year.csv', index=False)
    plot_dir = directory / 'comparison_plots'
    plot_dir.mkdir(exist_ok=True)
    images = []

    def save(fig, name, caption):
        fig.tight_layout()
        path = plot_dir / f'{name}.png'
        fig.savefig(path, dpi=125, bbox_inches='tight')
        plt.close(fig)
        encoded = base64.b64encode(path.read_bytes()).decode()
        images.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')

    horizons = sorted(test.horizon.unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 9), squeeze=False)
    for ax, h in zip(axes.flat, horizons):
        f = common[common.horizon == h].sort_values('target_date')
        actual = f.drop_duplicates('target_date')
        ax.plot(pd.to_datetime(actual.target_date), actual.actual_price, color='black', label='Actual', linewidth=1)
        for model in MODELS:
            m = f[f.model == model]
            ax.plot(pd.to_datetime(m.target_date), m.point_price, color=COLORS[model], label=model, alpha=.75)
        ax.set(title=f'{h}-session held-out prices', ylabel='GLD proxy (unverified units)')
        ax.legend(ncol=4)
    save(fig, 'actual_predictions', 'Actual versus predicted prices on identical eligible test pairs.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['mae', 'rmse']):
        pivot = scores.pivot(index='horizon', columns='model', values=metric).reindex(columns=MODELS)
        pivot.plot.bar(ax=ax, color=[COLORS[m] for m in MODELS], rot=0)
        ax.set(title=f'{metric.upper()} — lower is better', xlabel='Observed-session horizon', ylabel='GLD proxy units')
    save(fig, 'accuracy', 'Price error metrics; these are not accuracy percentages.')

    fig, axes = plt.subplots(len(horizons), 2, figsize=(13, 10))
    for row, h in enumerate(horizons):
        groups = []
        for model in MODELS:
            f = common[(common.horizon == h) & (common.model == model)].sort_values('target_date')
            errors = f.actual_price - f.point_price
            groups.append(errors.to_numpy())
            axes[row, 0].plot(pd.to_datetime(f.target_date), errors, label=model, color=COLORS[model], alpha=.7)
        axes[row, 0].axhline(0, color='black', lw=.8)
        axes[row, 0].set(title=f'{h}-session signed errors: actual − predicted', ylabel='GLD proxy units')
        axes[row, 0].legend(ncol=3)
        boxes = axes[row, 1].boxplot(groups, tick_labels=MODELS, patch_artist=True)
        for box, model in zip(boxes['boxes'], MODELS):
            box.set_facecolor(COLORS[model])
        axes[row, 1].axhline(0, color='black', lw=.8)
        axes[row, 1].set(title=f'{h}-session observed error distribution', ylabel='GLD proxy units')
    save(fig, 'errors', 'Historical error ranges describe observed errors; they are not forward prediction intervals.')

    for h in horizons:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
        for ax, model in zip(axes, ['ARIMA', 'GARCH']):
            f = common[(common.horizon == h) & (common.model == model)].sort_values('target_date')
            x = pd.to_datetime(f.target_date)
            for level, alpha in [(95, .15), (80, .3)]:
                ax.fill_between(x, f[f'lower{level}'], f[f'upper{level}'], color=COLORS[model], alpha=alpha, label=f'{level}% PI')
            ax.plot(x, f.actual_price, color='black', label='Actual', lw=.8)
            ax.plot(x, f.point_price, color=COLORS[model], label='Point')
            ax.set(title=f'{model}: {h}-session test prediction intervals', ylabel='GLD proxy units')
            ax.legend(ncol=4)
        save(fig, f'intervals_{h}', f'{h}-session prediction intervals: matching ARIMA/GARCH axes, fitted-parameter Gaussian simulation.')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, level in enumerate((80, 95)):
        for col, metric in enumerate((f'coverage{level}', f'width{level}', f'interval_score{level}')):
            scores.pivot(index='horizon', columns='model', values=metric).reindex(columns=MODELS).plot.bar(
                ax=axes[row, col], color=[COLORS[m] for m in MODELS], rot=0)
            axes[row, col].set_title(metric)
            if col == 0:
                axes[row, col].axhline(level / 100, color='black', ls='--', label='Nominal')
                axes[row, col].set_ylim(0, 1.05)
            axes[row, col].set_xlabel('Observed-session horizon')
    save(fig, 'uncertainty_scores', 'Coverage compared with nominal dashed lines, interval widths, and interval scores (lower score is better).')

    if not final.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, model in zip(axes, ['ARIMA', 'GARCH']):
            f = final[(final.model == model) & final.fit_status.eq('ok')]
            for level, alpha in [(95, .15), (80, .3)]:
                ax.fill_between(f.horizon, f[f'lower{level}'], f[f'upper{level}'], color=COLORS[model], alpha=alpha, label=f'{level}% PI')
            ax.plot(f.horizon, f.point_price, color=COLORS[model], label='Point')
            ax.set(title=f'{model}: historical as-of {final.origin.iloc[0]}', xlabel='Future observed-session step', ylabel='GLD proxy units', xticks=[1, 5, 10, 15, 20])
            ax.legend()
        save(fig, 'final_forecasts', 'Historical as-of forecasts; future calendar dates and measured accuracy are unavailable.')

    if (directory / 'training_residuals.csv').exists():
        residuals = pd.read_csv(directory / 'training_residuals.csv', index_col=0, parse_dates=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        residuals.return_percent.plot(ax=axes[0, 0], title='Initial-training percentage log returns', lw=.6)
        residuals.return_percent.rolling(20).std().plot(ax=axes[0, 1], title='20-observation return volatility')
        from statsmodels.graphics.tsaplots import plot_acf
        for ax, name in zip(axes[1], ['ARIMA', 'GARCH']):
            if name in residuals:
                plot_acf(residuals[name], ax=ax, lags=20, title=f'{name} training residual ACF')
        save(fig, 'diagnostics', 'Initial-training diagnostics only. GARCH residuals are standardized.')

    message = interpretation(scores, paired)
    quality = json.loads((directory / 'quality_report.json').read_text())
    limitations = ('GLD gold-price proxy; instrument, currency, close/adjusted-close policy and licensing remain unverified. '
        'Values are not established spot gold prices per ounce. Sessions are observed rows, with no filling and no validated exchange calendar. '
        'Gaussian innovations and fitted-parameter intervals omit parameter uncertainty and can miss heavy tails. '
        'The systems have different mean specifications, so differences cannot be attributed solely to GARCH volatility. '
        'QLIKE uses percentage-return variance (floor 1e-12) and the common zero-mean squared daily return proxy; this is a noisy approximation, not realized intraday variance. '
        'Bootstrap intervals use paired moving blocks; 20 sessions is primary, 10 and 40 provide sensitivity. '
        'Bootstrap rows with fewer than two requested blocks are flagged limited_sample and cannot support a winner claim. '
        'Yearly results are descriptive and do not retune models. Historical error ranges and forward prediction intervals have different meanings.')
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>GLD forecasting comparison</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#202830}}img{{width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #ddd}}.table{{overflow:auto}}figure{{margin:36px 0}}p{{line-height:1.6}}</style>
<h1>GLD proxy: ARIMA, GARCH and naive comparison</h1>
<p>Run {html.escape(directory.name)}. Test targets: {test.target_date.min()} through {test.target_date.max()}. Input cutoff: {quality['end']}. Split counts: {quality['split_counts']}. Units unverified.</p>
<h2>Findings</h2><p>{html.escape(message)}</p><h2>Protocol and limitations</h2><p>{limitations}</p>
<h2>Exact accuracy, empirical error ranges and uncertainty metrics</h2><div class="table">{scores.to_html(index=False, float_format=lambda x: f'{x:.6f}')}</div>
<details><summary>Paired inference and block-length sensitivity — exact values</summary><div class="table">{paired.to_html(index=False, float_format=lambda x: f'{x:.6f}')}</div></details>
{''.join(images)}<h2>Final historical as-of forecasts: highlighted steps</h2><p>These forecasts are separate from measured test accuracy.</p>
<div class="table">{final[final.horizon.isin([1,5,20])].drop(columns=['fit_details'], errors='ignore').to_html(index=False) if not final.empty else 'Not yet generated.'}</div>
{extension_section(directory)}
</html>'''
    (directory / 'comparison.html').write_text(body)
    (directory / 'report.html').write_text(body)
    (directory / 'interpretation.txt').write_text(message + '\n\n' + limitations + '\n')
    return scores
