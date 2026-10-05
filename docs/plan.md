# Gold forecasting architecture plan

Status: implemented; see [implementation and verification notes](implementation.md) for execution evidence and limitations.
Prepared: October 4, 2026.

## 1. Objective and requirements

Build a reproducible Python analysis that forecasts a gold-price proxy using ARIMA and a mean model with GARCH volatility, evaluates both on unseen observations, and explains which performs better for price prediction and uncertainty estimation. A finding that neither beats a simple baseline is a valid outcome.

### Project evidence recorded during planning

- At planning time, the only existing project file was `gold_data_2015_25.csv`; no application, notebook, dependency configuration, or test suite existed yet.
- The CSV contains 2,666 observations from January 2, 2015 through August 14, 2025, with columns `Date`, `SPX`, `GLD`, `USO`, `SLV`, and `EUR/USD`.
- Inspection found ascending, unique dates, no weekend observations, no blank cells, and finite positive numeric values. This does not establish that every expected trading session is present.
- `GLD` ranges from 100.50 to approximately 316.29. Its provider, currency, price field, adjustment policy, and licensing are undocumented.

### Functional requirements

1. Preserve the source CSV and generate a data-quality report before modeling.
2. Use `GLD` as the provisional target. Label results as a GLD gold-price proxy until provenance is verified; do not claim that these values are spot gold prices per ounce. Confirm the instrument, units, close/adjusted-close definition, and trading calendar before interpreting results economically.
3. Forecast 1, 5, and 20 observed trading sessions ahead, with 1-session price MAE as the primary comparison criterion.
4. Produce price point forecasts, 80% and 95% prediction intervals, and one-session return variance forecasts.
5. Compare ARIMA, a constant-mean GARCH model, and a last-observed-price baseline on identical forecast origins and targets.
6. Use chronological validation and an untouched final evaluation period. Prevent future observations from entering preprocessing, tuning, or individual forecasts.
7. Save model settings, forecast records, accuracy tables, diagnostics, plots, and an explanatory report.
8. After evaluation, refit the frozen model specifications on all available observations and issue forecasts explicitly dated as of the last data observation.
9. Produce a separate, self-contained comparison report containing both ARIMA and GARCH results, with plots comparing accuracy, observed error ranges, and prediction intervals at each forecast horizon.

### Scope and defaults

- First release: local batch analysis with reusable modules and one presentation notebook; no web application or deployment.
- Use only `GLD` for modeling. Preserve the other columns but exclude them from the initial experiment; their future values would otherwise require separate forecasts or lagged availability rules.
- Do not download fresh data automatically. The supplied data ends more than a year before this plan, so forecasts from it are historical as-of forecasts, not current October 2026 predictions.
- Work in observed-session steps. Assign future calendar dates only after validating an exchange calendar; never treat generic weekdays as verified trading sessions.
- No trading strategy, trading execution, causal claims, or guaranteed forecast accuracy is in scope.

## 2. Modeling design and fair comparison

ARIMA specifies a conditional mean with autoregressive, differencing, and moving-average terms. GARCH specifies changing conditional variance and needs a mean specification to generate return or price forecasts. Their model components are therefore not interchangeable. See the official [statsmodels ARIMA reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) and [arch forecasting documentation](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).

### Common representation

- Set `x_t = log(GLD_t)` and percentage log returns `r_t = 100 * (x_t - x_(t-1))`.
- Use ARIMA `(p, 0, q)` on returns, equivalent to an integrated ARIMA `(p, 1, q)` mean specification on log prices, with appropriately mapped drift and scaling. Fix differencing by this representation; do not blindly difference returns again.
- Use training-only ADF/KPSS, ACF/PACF, return plots, and ARCH-effect diagnostics to assess assumptions, not to choose parameters using held-out data.
- Recover prices from coherent future return paths: `P_(t+h) = P_t * exp(sum(r_(t+j), j=1..h) / 100)`.
- Use the median simulated price as the primary point estimate for every model, consistent with MAE. Report RMSE as a secondary metric on these same forecasts; do not call an exponentiated mean log price an expected price.

### Required model set

| Model | Mean specification | Variance specification | Purpose |
| --- | --- | --- | --- |
| Naive baseline | Zero future log return | Training-window constant return variance | Last observed price; reference intervals |
| ARIMA | Return ARMA with `p,q` in `{0,1,2,3}`, including a constant | Constant Gaussian innovation variance | Forecast serial dependence in returns |
| GARCH | Estimated constant return mean | GARCH(1,1), Gaussian innovations | Forecast changing return volatility |

Use Gaussian innovations initially so the distribution choice does not confound the volatility comparison. Select the ARIMA order using chronological validation performance; AIC/BIC are diagnostics or prescreening aids, not evidence of out-of-sample superiority. Keep GARCH(1,1) fixed in the first experiment to limit tuning and fitting cost.

The ARIMA/GARCH comparison is a comparison of complete forecasting systems; their mean specifications differ. Add a controlled ARIMA–GARCH extension after the required experiment if attribution is needed: retain the selected ARIMA mean, fit GARCH to its training residuals, and simulate innovations through the same mean recursion. Label this as a two-stage fit, not joint maximum likelihood. It isolates the practical benefit of changing the variance specification but is not required for the initial deliverable.

### Forecast contract

- Every model adapter accepts only data through the forecast origin, horizon, and explicit random seed; returns aligned point forecasts, interval bounds, variance forecasts, and fit status.
- Generate at least 10,000 conditional return paths per origin using recorded seeds. Propagate mean dynamics and, for GARCH, evolving conditional variances along each path.
- Form multi-session intervals from cumulative path returns. Do not sum marginal interval endpoints or use square-root-of-time scaling for GARCH price intervals.
- Keep percent-return variances and decimal-return variances distinct; converting the former to the latter requires dividing by 10,000.
- Distinguish forecast return variance from innovation variance when mean dynamics are present; the [arch forecast interface](https://arch.readthedocs.io/en/stable/univariate/generated/arch.univariate.base.ARCHModelForecast.html) exposes both.
- Record convergence warnings and invalid forecasts. Do not silently substitute another model or remove difficult observations.

## 3. Evaluation and winner selection

### Chronological protocol

1. Initial training: 2015–2021, 1,760 price observations.
2. Validation: 2022–2023, 501 price observations. Run expanding-window forecasts to choose the ARIMA order and freeze all modeling and scoring settings.
3. Final test: 2024–August 14, 2025, 405 price observations. Evaluate once after settings are frozen.
4. At each origin, refit parameters using only observations available at that origin. Previously observed validation/test values may enter later expanding training windows, but must not inform hyperparameter selection on the test period.
5. Start forecasts at the session preceding each evaluation segment. For each horizon, score only origins whose complete target lies within that segment; never tune on validation-origin forecasts with test-period targets.
6. Compare models on matching origin/horizon pairs. Report attempted forecasts, failures, and coverage; a model with failures cannot appear superior merely because it omitted hard cases. Publish common-pair results with explicit coverage and require complete valid coverage for an unqualified winner.

### Metrics and interpretation

- **Primary:** one-session price MAE. Freeze this choice before evaluating the test period.
- **Secondary:** price RMSE and MAE for 5 and 20 sessions, plus relative improvement versus the naive baseline for each horizon. Do not combine horizons into an unexplained overall score.
- **Uncertainty:** empirical 80%/95% interval coverage, average width, and interval score at each nominal level. Narrow intervals alone are not better; check calibration and interval score together.
- **Variance:** one-session QLIKE, `log(v_t) + r_t^2 / v_t`, using the same zero-mean squared-return proxy for every model and a documented positive numerical floor. State that squared daily returns are a noisy volatility proxy and the common zero-mean assumption is an approximation; the dataset has no realized intraday variance.
- Apply a paired moving-block bootstrap to test-period loss differences, preserving temporal dependence. Use a fixed default block length of 20 sessions and report sensitivity to 10 and 40; save seeds and confidence intervals. This is especially relevant to overlapping multi-session forecasts.
- Name separate leaders for price accuracy and uncertainty quality. If uncertainty in the primary loss difference includes zero, describe the evidence as inconclusive rather than declaring a definitive winner.
- A model can rank first among ARIMA/GARCH and still fail to beat the naive baseline. State both conclusions.

After evaluation, retain the untouched test results, refit the frozen specifications on the complete dataset, and export the next 20 session-step forecasts, highlighting steps 1, 5, and 20. This refit must not overwrite backtest artifacts.

### Separate comparison output

Generate `outputs/<run_id>/comparison.html` as a standalone report with embedded plots, viewable without running the notebook or a server. Include both models together, the naive baseline for context, the evaluation dates, units, sample counts, and a concise interpretation of which model performs better and where evidence is inconclusive.

Required comparison views:

- **Actual versus predicted prices:** overlay held-out actual prices and both models' point forecasts, using a separate panel for each of the 1-, 5-, and 20-session horizons.
- **Accuracy:** grouped MAE and RMSE bar charts by model and horizon, accompanied by exact metric values and improvement versus the naive baseline. Label lower values as better; do not invent an accuracy percentage from regression errors.
- **Observed error range:** plot signed errors (`actual - predicted`) over time with a zero reference line and show side-by-side error box plots by model and horizon. Include a table of minimum, maximum, median, and 5th/95th percentile errors so the empirical range and tails are explicit.
- **Forecast uncertainty:** show separate ARIMA and GARCH panels with matching axes, actual prices, point forecasts, and shaded 80%/95% prediction intervals. Compare empirical coverage against nominal coverage, average interval width, and interval score using plots and a summary table.
- **Final forecasts:** show both models' next 20 session-step forecasts and prediction bands in a clearly separated as-of section. Do not attach measured accuracy to future observations whose actual values are unavailable.

Distinguish historical error ranges from forward-looking prediction intervals in labels and explanatory text. Use consistent model colors, readable legends, shared comparison scales, and identical eligible test pairs. Export plotted data to `comparison_metrics.csv` and `comparison_forecasts.csv` and save plots under `comparison_plots/` within the same run directory. Compute the report from persisted forecast records so tables and plots agree with the main evaluation.

## 4. Proposed changes and important files

Only this plan is created during the architecture phase. Implement the following files in a later phase:

| File or directory | Role |
| --- | --- |
| `gold_data_2015_25.csv` | Existing immutable input; record SHA-256 in each run |
| `docs/plan.md` | This architecture and evaluation contract |
| `README.md` | Setup, commands, data limitations, and result interpretation |
| `pyproject.toml` and dependency lockfile | Supported Python version and reproducible dependencies |
| `configs/default.toml` | Target, split dates, horizons, model grid, simulation count, seeds, output settings |
| `src/gold_forecast/data.py` | CSV validation, date handling, transformations, split construction |
| `src/gold_forecast/models.py` | Baseline, ARIMA, and GARCH adapters; forecast and fit-status contract |
| `src/gold_forecast/backtest.py` | Chronological tuning and frozen expanding-window evaluation |
| `src/gold_forecast/metrics.py` | Price, interval, variance metrics and paired loss analysis |
| `src/gold_forecast/report.py` | Tables, plots, diagnostics, and report generation |
| `src/gold_forecast/cli.py` | Commands to validate, tune, evaluate, and forecast |
| `notebooks/gold_price_forecasting.ipynb` | Narrative analysis using package functions and saved outputs |
| `tests/` | Data, alignment, leakage, modeling, metrics, and pipeline verification |
| `outputs/<run_id>/` | Manifest, quality report, forecasts, scores, diagnostics, plots, and final report |
| `outputs/<run_id>/comparison.html` | Separate standalone ARIMA/GARCH results report with embedded accuracy, error-range, and prediction-interval plots |
| `outputs/<run_id>/comparison_metrics.csv` | Model/horizon accuracy, error summaries, interval metrics, and evaluated sample counts |
| `outputs/<run_id>/comparison_forecasts.csv` | Aligned forecasts, actuals, signed errors, interval bounds, and phase labels supporting comparison plots |
| `outputs/<run_id>/comparison_plots/` | Exported comparison figures for reuse outside the report |
| `.gitignore` | Exclude environments, caches, and generated run artifacts |

Use pandas/NumPy for data, statsmodels for ARIMA and diagnostics, arch for GARCH, SciPy for statistical support, Matplotlib for plots, pytest for tests, and Jupyter for presentation. Resolve and lock compatible released versions during implementation; do not depend on an unpinned development release.

Persist forecast rows with `run_id`, `phase`, `model`, `origin`, `training_end`, `horizon`, `target_date` when known, `actual_price` when observed, `point_price`, interval bounds, return variance with units, and fit status. The run manifest records input checksum, configuration, package versions, seeds, selected parameters, and execution timestamp. Plot held-out actuals against forecasts, interval bands, errors by horizon, returns/volatility, and residual diagnostics.

Implementation order: establish data contract and baseline; add model adapters; implement validation and evaluation; add diagnostics/reporting; execute verification and publish the notebook plus saved results.

## 5. Risks and design concerns

- **Target ambiguity:** the column name alone cannot verify spot gold or ETF semantics. Resolve provenance before presenting dollar-denominated economic conclusions; changing to spot gold requires a new dataset and rerun.
- **Stale observations:** the initial forecast is anchored to August 14, 2025. A current forecast requires a separately documented data refresh.
- **Calendar and missing-session issues:** inspect unexpected gaps against a verified calendar. Fail or explicitly document treatment of missing sessions; do not forward-fill prices to manufacture zero returns. Weekends and holidays are not missing observations.
- **Leakage:** full-sample normalization, selection from test results, backward filling, or using contemporaneous external assets before availability would invalidate the comparison.
- **Structural changes:** market regimes can change. Report performance by year as descriptive analysis without retuning on test slices.
- **Model limitations:** financial return predictability may be weak; GARCH may improve intervals without improving the price mean. Preserve that distinction in conclusions.
- **Numerical stability:** check finite forecasts, optimizer convergence, positive GARCH variance parameters, and persistence `alpha + beta < 1` for the stationary specification. Flag boundary fits and record any predefined training-only retry policy.
- **Distribution and parameter uncertainty:** Gaussian innovations can underrepresent heavy tails; fitted-parameter simulation intervals omit parameter uncertainty. Evaluate observed coverage and label this limitation. Student-t innovations and parameter bootstrap are later extensions, not post-test fixes.
- **Simulation/runtime:** rolling fits and path simulation can be expensive. Cache validated deterministic artifacts, measure runtime, and check Monte Carlo stability before reducing precision or altering the frozen evaluation schedule.
- **Reproducibility:** keep the raw CSV immutable, pin dependencies, persist settings and seeds, and distinguish validation, test, and final-refit outputs.

## 6. Testing and verification steps

### Data and transformations

- Verify required columns, parseable dates, strict chronology, uniqueness, positive finite target values, and expected split counts; add fixtures for malformed dates, duplicates, missing values, and nonpositive prices.
- Verify log-return and price-path reconstruction on a hand-computed sequence, including percent/decimal variance conversion.
- Verify missing-session handling once the calendar is established and that no synthetic weekend/holiday observations are introduced.

### Chronology and leakage

- Assert `training_end <= origin < target_date` for scored forecasts and exact session offsets for every horizon.
- Modify all observations after a chosen origin and confirm that its model fit and forecasts do not change.
- Verify that validation forecasts cannot score test-period targets, tuning never reads test observations, and all models receive identical origin histories.

### Models and statistics

- On synthetic random-walk data, verify last-price forecasts and sensible ARIMA behavior without imposing unrealistic accuracy thresholds.
- On seeded synthetic GARCH data, verify positive finite variance forecasts and the expected multi-step variance recursion within numerical tolerance.
- Cross-check a small ARIMA/GARCH forecast against library outputs and verify ordered 80%/95% intervals, with wider nominal intervals containing narrower ones.
- Verify MAE, RMSE, interval scores, and QLIKE with hand-calculated examples, including invalid variance handling.
- Verify that failure status survives aggregation and that the paired bootstrap resamples paired chronological loss blocks.

### End-to-end acceptance

1. Run a small seeded pipeline smoke test from a clean environment and confirm all required outputs and manifest fields exist.
2. Run the complete validation and frozen test workflow. Require finite valid forecasts for every eligible origin/horizon before making an unqualified winner claim; disclose failures otherwise.
3. Rerun with identical dependencies and seeds; require equal splits/model selections and metrics within declared numerical tolerances.
4. Check simulation quantile stability on representative origins using additional paths and a second seed before freezing the simulation count.
5. Inspect residual autocorrelation and squared-residual dependence (Ljung–Box/ARCH diagnostics), GARCH standardized residuals, coverage, and plots. Diagnostics inform limitations, not automatic test-set retuning.
6. Confirm the report identifies the target and units, data cutoff, model assumptions, primary winner or inconclusive result, naive comparison, horizon-specific results, uncertainty results, and limitations.
7. Execute the presentation notebook from a clean kernel and ensure its numbers match persisted outputs; verify final forecasts are labeled as historical as-of forecasts and separate from test scores.
8. Open the standalone comparison report without the notebook or a server and verify all embedded plots render. Check that both models appear at every horizon, plotted values match exported CSVs, error signs and quantiles are correct, interval coverage matches recorded bounds, and final forecasts are excluded from measured accuracy. Visually inspect legends, axis units, consistent scales, and the distinction between observed error ranges and prediction intervals.

Success means a reproducible, leakage-controlled comparison and honest forecast reporting—not a predetermined winning model or accuracy threshold.

## 7. User-requested extension: calendar-year 2026 plot

After the initial experiment, the user requested a final plot including the observed 2024–2025 trend and forecasts through 2026. Implemented as a separate `project-2026` command and `projection_2026/` artifacts, appended to the notebook and standalone reports. Future dates use an explicitly assumed published NYSE holiday schedule; dataset provenance and the historical calendar remain unverified. Observed prices stop on August 14, 2025. Everything later is a historical as-of projection, with no claimed 2026 actuals or accuracy. The original frozen 1/5/20-session evaluation remains separate.

## 8. User-requested extension: AR reference

Add a fixed AR(1) with a constant on percentage log returns and constant Gaussian
innovation variance. Implement it as return ARIMA(1,0,0), using the same exact
joint conditional covariance and path-to-price reconstruction. The lag is fixed
without tuning, and the last-price baseline remains available. Append AR's seed
offset as 3 so the original Naive/ARIMA/GARCH offsets do not change.

New runs freeze all four model names. Include AR in validation, test, final
forecasts, training diagnostics, precision audits, plots, metrics, paired
inference, and the optional 2026 projection. Compare all four models on common
valid pairs; AR failures must reduce coverage explicitly.

Existing three-model experiments can be extended into a new directory using
`scripts/add_ar_reference.py`. Preserve their original forecasts and selected
ARIMA order; record source provenance and the current environment. Label AR
comparisons from this extension as retrospective/exploratory because the model
was added after the original test evaluation. Never overwrite the source run.
