# GLD proxy forecasting

A reproducible local comparison of a last-price baseline, fixed return AR(1) reference, return ARIMA and constant-mean GARCH(1,1), implementing [the architecture plan](docs/plan.md). The primary endpoint is one-observed-session price MAE. Five- and twenty-session errors, interval calibration and scores, and one-session QLIKE are separate endpoints.

## AR reference comparison

New runs include **AR**, a fixed AR(1) with a constant on percentage log returns,
Gaussian innovations, and the same expanding histories and conditional-path
forecast contract. Its lag is fixed at one, independent of ARIMA order selection.
The existing Naive last-price baseline remains the reference for percentage
improvement. AR appears in accuracy, error, interval, bootstrap, final-forecast,
and optional 2026 projection outputs. Old three-model runs remain readable.

Open `outputs/full-20261004-ar/comparison.html` for the expanded saved comparison.
All 4,768 eligible test rows are valid. AR MAE is 2.0183, 4.2473, and 7.9119 at
1, 5, and 20 sessions respectively. The 37-test suite passes. This reference was
added after the original evaluation, so AR comparisons are retrospective and
exploratory. The original three-model forecasts and order selection are preserved.

To add AR to another completed three-model run without repeating ARIMA tuning:

```sh
uv run python scripts/add_ar_reference.py outputs/my-analysis outputs/my-analysis-ar
uv run python scripts/verify_run.py outputs/my-analysis-ar
```

The destination must not exist. The extension records the source manifest and
current environment, retains original model forecasts, and regenerates reports.
If the source includes the optional 2026 projection, that view is regenerated
with AR too. Use the new directory as `GOLD_RUN_DIR` when running the notebook.

## Original three-model verified results

The complete experiment and an independent repeat selected return ARIMA `(3,0,1)`. All 3,576 eligible test forecast rows were valid. One-session MAE was 2.0169 for GARCH, 2.0241 for ARIMA and 2.0217 for naive, in unverified GLD proxy units. The ARIMA–GARCH primary comparison was statistically inconclusive. GARCH slightly beat naive at one session; ARIMA had lower MAE and RMSE at 5 and 20 sessions. GARCH had better one-session interval scores, while ARIMA had better 5/20-session interval scores.

The 24-test suite passed, the notebook executed, and 24 CSV artifacts matched between complete runs. The repeat run took about 42 minutes with four candidate workers. Open `outputs/full-20261004/comparison.html` for the local standalone report. See [verification details and limitations](docs/implementation.md).

## Setup and commands

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). `uv.lock` pins the resolved release dependencies, including development/notebook dependencies.

```sh
uv sync --locked --extra dev
uv run gold-forecast validate
uv run pytest -q
uv run gold-forecast run --run-id my-analysis --workers 4
```

The full run performs 16 × 501 ARIMA validation fits, then expanding-window test fits and 10,000 conditional paths per origin. It can take substantial time. `--workers` runs independent order candidates in separate processes; it does not change seeds or observations. Outputs are local and ignored by Git. The raw CSV is read only.

For separately invocable stages:

```sh
uv run gold-forecast tune --run-id staged --workers 4
uv run gold-forecast evaluate --run-id staged
uv run gold-forecast forecast --run-id staged
uv run gold-forecast report --run-id staged
```

Use `--config path/to/config.toml` consistently for every stage. Run IDs cannot be reused for tuning; evaluation refuses to overwrite existing test records. Existing-run commands reject changed input checksums or configuration. An interrupted tune requires a new run ID; partial candidate CSVs remain available for inspection. The presentation notebook reads a completed run; set `GOLD_RUN_DIR` to choose one, or it uses the most recently completed local run.

```sh
GOLD_RUN_DIR="$PWD/outputs/my-analysis" uv run jupyter nbconvert \
  --to notebook --execute --inplace notebooks/gold_price_forecasting.ipynb
```

In environments with restricted cache directories, set `UV_CACHE_DIR`, `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable temporary directories. Notebook execution requires permission to open a local kernel connection.

## Data contract and interpretation

The supplied file has 2,666 observations, from January 2, 2015 through August 14, 2025. `GLD` is a **provisional gold-price proxy**, not verified spot gold per ounce. Provider, instrument, currency, close/adjusted-close policy and license are unknown. Do not make currency-denominated economic interpretations until these are established. Other columns are validated and preserved in the input but excluded from modeling.

The quality report records chronology, positivity, finite values, split counts, checksum, weekends and calendar gaps greater than four days. There is a five-calendar-day gap ending April 22, 2025. The exchange calendar is unverified: no claim of complete trading sessions is made. No dates or prices are filled. Horizons count supplied observations, and final forecasts use session steps without fabricated calendar dates. All final forecasts are historical **as of August 14, 2025**, not current forecasts.

Training ends in 2021 (1,760 prices); validation is 2022–2023 (501); test is 2024–August 2025 (405). Forecasts begin at the observation preceding each segment. A horizon is eligible only if its target lies inside that segment. Previous test observations can enter subsequent fits, but cannot affect order selection. Validation selects the order from `{0,1,2,3}²` using one-session price MAE, requiring complete one-session coverage. Ties resolve by lower p then q. All scoring and simulation settings are frozen before test evaluation.

## Important modeling decisions

- Models use percentage log returns, `100 * diff(log(GLD))`. Return ARIMA `(p,0,q)` includes a constant; GARCH has a constant mean and Gaussian innovations. This compares complete systems with different means, not an isolated volatility treatment.
- ARIMA paths use the exact Gaussian joint conditional forecast covariance from the filtered state, including off-diagonal dependence. GARCH paths recursively evolve conditional variance after each simulated innovation. Prices are reconstructed from cumulative returns.
- Gaussian draws are antithetically paired to reduce median simulation noise. The primary estimate is the simulated price median. The baseline point is exactly the last observed price (its population median). Intervals are empirical path quantiles, not sums of endpoints. Even path counts are recommended.
- Return variance and innovation variance are different for dynamic means. Exported variance is the marginal return forecast variance at each step. QLIKE only scores the one-session variance; percentage-squared units convert to decimal-squared by dividing by 10,000.
- ARIMA gets one predefined optimizer retry (300 iterations, then 1,000 initialized from the preceding fit). Warnings, parameters and seeds are persisted. Nonconverged fits and GARCH persistence at or above one fail explicitly, without replacement.
- Rankings use identical valid model/origin/horizon pairs. Tables also include attempted counts, failed counts and valid coverage. Failed rows are retained in the original forecast CSV. Incomplete coverage prohibits an unqualified winner claim.
- Paired moving-block bootstrap intervals operate on chronologically ordered paired losses. The primary block is 20 sessions, with 10/40 sensitivity. Loss differences are model A minus model B; negative favors A. Intervals containing zero imply inconclusive evidence. Fewer than two requested blocks is flagged as limited-sample inference; effective block lengths are exported and cannot support a winner claim. Bootstrap results cover price MAE and both interval scores separately by horizon.
- QLIKE uses `log(v) + r²/v`, with a positive `1e-12` floor after rejecting invalid variance. Daily squared returns are a noisy volatility proxy and the shared zero-mean assumption is approximate.
- Gaussian intervals omit parameter uncertainty. Residual diagnostics and descriptive yearly results do not trigger test-period retuning. The optional two-stage ARIMA–GARCH extension is outside this release.

## Outputs

`outputs/<run_id>/comparison.html` is self-contained: PNG plots are embedded, with no server, notebook or network needed. It includes actual/predicted prices; MAE/RMSE; signed error histories and box plots; empirical error quantiles; model interval panels on matching axes; coverage, width and interval-score comparisons; and a separate final as-of forecast section. Historical errors are not prediction intervals, and future forecasts receive no accuracy score.

Other artifacts include:

- `manifest.json`, `quality_report.json`, `frozen.json`: input identity, dependencies, configuration, seeds and selected order.
- `validation_arima_*.csv`, `tuning_scores.csv`, `validation_forecasts.csv`: complete tuning evidence, including failures.
- `test_forecasts.csv`, `final_forecasts.csv`: distinct backtest and refit artifacts, fit details and variance units.
- `scores.csv`, `scores_by_year.csv`, `paired_bootstrap.csv`: overall and descriptive annual scores, and paired inference.
- `diagnostics.json`, `training_residuals.csv`, `simulation_stability.csv`: training-only diagnostics and a precision audit at two initial-training origins, using a second seed and twice as many paths. The preselection ARIMA audit uses `(0,0)` and is not evidence about every candidate order.
- `comparison_metrics.csv`, `comparison_forecasts.csv`, `comparison_plots/`: exact plotted data, errors and reusable plots. The former has one row per model/horizon; the latter contains common test pairs and separate final forecasts.
- `report.html`, `interpretation.txt`: explanatory results; `report.html` contains the same standalone content as `comparison.html`.

## Verification

Tests cover malformed input, transforms, segment eligibility, future-data mutation, identical histories, library cross-checks of ARIMA joint covariance and GARCH variance recursion, interval ordering, variance units, hand-calculated metrics, paired block resampling, failure coverage, artifact generation, overwrite guards and seeded smoke-run reproducibility. The smoke test uses synthetic data and never claims predictive superiority.

Audit a completed run with `uv run python scripts/verify_run.py outputs/my-analysis`. Add `--compare outputs/repeat-analysis` to verify independently rerun tuning, forecasts and metrics within the declared tolerances. This audit checks session offsets, phase boundaries, variance units, interval nesting, CSV/report agreement and embedded image integrity, and saves `verification.json`.

See [implementation and verification notes](docs/implementation.md) for run results and remaining limitations. Library references: [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) and [arch forecasting](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).

## Plot the 2024–2025 trend and forecasts through 2026

```sh
uv run gold-forecast project-2026 --run-id full-20261004
```

This adds a final section to `comparison.html` and `report.html`, and the notebook ends with the same plot. It shows observed prices from January 2024 through August 14, 2025, followed by separate ARIMA and GARCH forecast medians and 80%/95% bands through December 31, 2026. No actual 2026 prices are available in the supplied data. The 347-step extension is an untested long-horizon extrapolation, not a new accuracy evaluation.

For this requested calendar-year view only, future dates **assume** the [published NYSE/NYSE Arca holiday schedule](https://ir.theice.com/press/news-details/2024/NYSE-Group-Announces-2025-2026-and-2027-Holiday-and-Early-Closings-Calendar/default.aspx), checked against the [2026 NYSE calendar](https://www.nyse.com/trade/hours-calendars). Weekends and full-day closures are excluded; early closes count as sessions. This does not verify the CSV's instrument or historical trading calendar. Exceptional closures and missing future observations are not modeled. The command deliberately supports only the supplied August 14, 2025 cutoff.

The optional extension saves `projection_2026/forecast_2024_2026.png` and `.svg`, `forecasts.csv`, `observations.csv`, and `metadata.json` inside the run directory. It refits the already selected specifications, uses 10,000 paths per model, and keeps the original test and 20-step forecast records separate. On this run, December 2026 median projections are approximately **349.5 (ARIMA)** and **330.0 (GARCH)** in unverified GLD proxy units; their wide bands are more informative than those point values alone.

## Independent testing review

See [the independent review](docs/testing-review.md) for the audit correction,
33-test verification, clean installation and notebook checks, and remaining limitations.

## Manual Smoketest
Manual Smoketest passed. The ipynb and all other scripts are running well manually. Data Reading is normal.
