# Implementation and verification

Implemented October 4, 2026. The architecture in [plan.md](plan.md) is realized as a reusable Python package, staged CLI, TOML configuration, dependency lockfile, tests, presentation notebook and standalone comparison reports.

## Design choices

The experiment uses percentage log returns throughout. ARIMA's Gaussian forecast covariance is built from the final filtered-state covariance and state transition, rather than treating future returns as independent. A test compares the full covariance against statsmodels conditional simulations. GARCH uses analytical marginal return variances for scoring and recursive pathwise innovation/variance updates for price quantiles; a synthetic GARCH test cross-checks the library and the analytical recursion.

Antithetic Gaussian paths preserve the intended innovation law and reduce Monte Carlo median noise. The baseline is fixed exactly at last price. All other points are pathwise price medians, and all intervals come from cumulative-return prices. Invalid fits remain explicit failed rows. ARIMA optimizer retries use a fixed training-only rule. GARCH persistence at or above one is rejected for this stationary specification.

The tuning worker receives only training and validation observations. It has no access to test values. All test models receive identical histories, with only eligible segment-local targets scored. Selection requires complete one-session validation coverage, and test ranking uses common pairs while retaining model-specific attempted/failure counts. No held-out result changes the specification.

The report recomputes metrics from persisted test records. Final forecasts are appended only to the plotting export with a distinct phase, never to the scoring input. The report embeds PNG images so it can be opened offline. The notebook independently recomputes scores and asserts agreement with the persisted score table.

## Limitations preserved from the plan

- Instrument, currency, adjustment policy, provider and license are unknown. No economic spot-gold interpretation is warranted.
- The exchange calendar is unverified. The data has a five-calendar-day gap ending April 22, 2025. Sessions are supplied observations; there is no forward filling or fabricated future date assignment.
- Final forecasts are historical as of August 14, 2025.
- The Gaussian models omit parameter uncertainty and may underrepresent tails. The comparison changes both mean and variance specifications.
- QLIKE uses noisy daily squared returns, not intraday realized variance.
- The optional two-stage ARIMA–GARCH extension is intentionally outside the required experiment.
- Dependency/platform changes can alter numerical optimization. Use the lockfile; reproducibility tolerances are `rtol=1e-8`, `atol=1e-10` on the same platform.

## Verification evidence

- `uv lock --check --offline`: passed; the project metadata matches `uv.lock`.
- `MPLCONFIGDIR=/tmp/gold-mpl .venv/bin/python -m pytest -q`: **24 passed**. Two library warnings are visible in the ARIMA cross-check (NumPy deprecation and an automatically corrected starting parameter); neither indicates a failed test.
- The synthetic pipeline was independently replayed with the same seeds; forecast records matched exactly.
- The presentation notebook executed successfully from a clean Jupyter kernel against the smoke artifacts and asserted agreement between recalculated and saved metrics.
- The standalone smoke HTML opened from a local `file:` URL in Edge. Embedded charts rendered without a notebook or server. All nine embedded PNGs also passed decoding/integrity checks, and the interval plot was visually inspected for shared scales, legends and nested bands.
- Short bootstrap samples are explicitly flagged when they contain fewer than two requested blocks. A regression test prevents treating this edge case as definitive winner evidence.

Full-data execution and independent repeat-run evidence follow below.

### Full experiment and reproducibility

Completed `full-20261004` and the independent `repeat-20261004` run using the complete 16-order grid and 10,000 paths per origin. The measured repeat run took 2,501 seconds (about 42 minutes) with four candidate workers on this machine. Both selected return ARIMA **(3,0,1)** with validation one-session MAE **1.191412** across all 501 origins. Orders (2,0,2), (2,0,3) and (3,0,2) had 1, 2 and 3 failed validation origins, respectively, and were ineligible for selection; their records remain saved.

Both runs produced **3,576 valid test rows with zero failures**: 405, 401 and 386 eligible origins per model for horizons 1, 5 and 20. Each also produced 60 final rows (20 steps × three models), all explicitly historical as-of August 14, 2025. The original input checksum remains `1ccca1ac6b9da9f66ee085b0d5b0e45994c0b2c00e292693424862af43b6890c`.

| Horizon | Naive MAE | ARIMA MAE | GARCH MAE |
| --- | ---: | ---: | ---: |
| 1 | 2.021728 | 2.024127 | 2.016892 |
| 5 | 4.325686 | 4.245438 | 4.281313 |
| 20 | 8.466865 | 7.917123 | 8.154635 |

Values are GLD proxy units, not established currency or spot-gold units. GARCH has the lowest primary MAE, but ARIMA versus GARCH is inconclusive: ARIMA-minus-GARCH MAE difference **0.007236**, primary paired 95% interval approximately **[-0.0088, 0.0263]**. GARCH improves one-session MAE over naive by only **0.239%**; the naive-minus-GARCH interval is approximately **[0.0027, 0.0080]**. This is a small observed improvement, not a promise of future accuracy. ARIMA has lower 5- and 20-session MAE and RMSE than GARCH and naive.

Uncertainty quality varies by horizon. GARCH has lower one-session interval scores, 80%/95% empirical coverage of **78.8%/93.8%**, and lower one-session QLIKE (**1.0253**, versus ARIMA **1.1274**). ARIMA has lower interval scores at 5 and 20 sessions; its 20-session 95% coverage is **95.3%**, compared with GARCH **90.2%**. Gaussian assumptions and the omission of parameter uncertainty remain limitations.

`python scripts/verify_run.py outputs/full-20261004 --compare outputs/repeat-20261004` passed. It compared **24 CSV artifacts**, including all candidate forecasts, tuning scores, validation/test/final records, metrics and paired inference, within `rtol=1e-8`, `atol=1e-10`. Frozen configurations and selected orders also matched. The verification checked interval ordering, exact observed-session offsets, phase boundaries, variance conversion, report/CSV agreement and all nine embedded PNGs. Evidence is saved in `outputs/full-20261004/verification.json`.

The final presentation notebook was executed from a clean kernel against the completed full-data artifacts and checked its recalculated scores against the saved table. Full-data accuracy and interval figures were visually inspected for labels, scales, legends and clipping. The completed full-data standalone HTML was also opened from a local file in Edge, and its embedded comparison plots rendered correctly. The reports and reusable plots are in `outputs/full-20261004/`; generated run folders are intentionally ignored by Git, while the executed notebook contains the visible results.

### Requested 2026 visualization extension

Added `gold-forecast project-2026 --run-id full-20261004`, a final notebook section, and a final embedded report figure. The chart includes supplied 2024–2025 observations and extends both frozen model specifications through December 31, 2026 (347 projected sessions, including 251 scheduled 2026 sessions). All post-cutoff values are labeled forecasts. The calendar mapping assumes the published NYSE holiday schedule; it does not establish the dataset's provenance. This separate extrapolation carries no measured 2026 accuracy claim and preserves the original backtest and final 20-session artifacts. Tests cover calendar closures, retained early-close sessions, cutoff restrictions, forecast-only rows, frozen settings, and artifact separation.
