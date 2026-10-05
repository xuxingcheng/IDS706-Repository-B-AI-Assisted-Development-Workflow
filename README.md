**Selected Option**

I selected Option 2: Rebuild Your Previous Project. I rebuilt my gold-price forecasting project to compare different time series models and make the results easier to understand.

**Purpose of the Project**

The purpose of this project is to compare ARIMA and GARCH for forecasting a gold-price proxy. I wanted to see how their prediction errors and uncertainty ranges differ when they use the same data. I also requested an AR(1) reference model and kept a simple baseline that predicts the last observed price. The project compares forecasts for 1, 5, and 20 observed trading sessions ahead. It produces a notebook and a separate report with predicted prices, error comparisons, and prediction intervals.

**Dataset**

The dataset contains 2,666 observations from January 2, 2015 through August 14, 2025. The project uses the GLD column as a gold-price proxy. The source, units, and price adjustment policy are not verified, so I do not describe these values as spot gold prices per ounce. The observed data stops in August 2025. Everything after that date, including the projections through 2026, is a forecast rather than an actual observation.

**How to Install**

To install the project, open a terminal in the repository folder and make sure Python and uv are installed. Run uv sync --locked --extra dev to install the package and its testing and notebook dependencies. The project allows Python 3.11 through 3.13, although the recorded clean-environment check used Python 3.13. The dependency lockfile keeps the installed package versions consistent.

**How to Run**

From the project folder, run uv run gold-forecast run --run-id my-analysis --workers 4. Use a new name instead of my-analysis if you have run it before. The full run may take around 40 minutes. When it finishes, open outputs/my-analysis/comparison.html to see the results and plots. You can also open notebooks/gold_price_forecasting.ipynb and run its cells to view the saved results.

**How to Test**

To run the automated tests, use uv run pytest -q. To audit a completed run, use uv run python scripts/verify_run.py outputs/my-analysis, replacing my-analysis with the actual run name. This checks the expected forecast records, actual values against the source data, forecast dates, interval bounds, variance units, and agreement between exported results and report data. The tests also cover invalid inputs, future-data leakage, model calculations, failure handling, and reproducibility.

**Manual Smoke Test and My Own Verification**

My manual smoke test passed. My recorded manual check was running the notebook and scripts and checking that the data loaded normally. This was my own check that the project could run and display its outputs, separate from the AI's written assessment. It did not prove that the models would correctly predict future gold prices. I am also keeping this personal check separate from the more detailed automated checks that the AI tester performed.

**AI-Assisted Verification Results**

The recorded AI-assisted verification found that all 37 automated tests passed after the AR reference was added. The earlier clean Python 3.13 installation check passed the 33 tests that existed at that stage, and notebook execution also succeeded. The expanded four-model run contained 4,768 valid test forecast rows with no failed test forecasts. Two saved original runs matched across 24 CSV artifacts. During the review, the tester compared those saved runs instead of repeating the roughly 42-minute full experiment. The AR extension preserved the original models' forecast records, and the updated notebook executed successfully.

**How Each AI Role Contributed**

I used ChatGPT/Codex for specific tasks in the architect, builder, and tester roles. The architect helped organize the requirements, evaluation method, important files, risks, and testing plan. It also updated the plan when I asked for a separate comparison report. The builder implemented the forecasting pipeline, tests, reports, and notebook, then added the requested plot showing the observed 2024–2025 trend and projections through 2026. The tester reviewed the implementation against the plan, ran checks, and fixed an audit problem that could allow missing or duplicated forecast records to pass verification. In the tester conversation, I also requested the AR reference, which was implemented and checked.

**Scope of AI Assistance**

The AI assistance included planning, implementation, testing, and documentation, so it was more than proofreading. My requests set the goal and the additions I wanted. I describe the assistance that is recorded rather than claiming that I wrote all the code or personally performed every technical check. The role transcripts are saved in docs/transcripts as xx134_architect.txt, xx134_builder.txt, and xx134_tester.txt. The builder transcript includes a labeled summary of some earlier exchanges rather than a complete word-for-word log. AI also assisted with this README revision, which happened after those role transcripts were saved.

**AI Recommendation I Accepted**

One AI recommendation I accepted was to use chronological training, validation, and test periods and evaluate price errors separately from uncertainty. I asked the builder to follow this plan. This made sense because a forecast should not use future observations to choose its settings. It also helped avoid treating a model with better uncertainty estimates as automatically better at predicting the price itself. Training uses 2015–2021, validation uses 2022–2023, and testing uses 2024 through August 2025.

**AI Recommendation I Changed**

One recommendation I changed was the original model set. The plan initially included ARIMA, GARCH, and a last-price baseline. After the testing review, I requested an AR model as another reference. The implementation uses a fixed AR(1), which provides a simpler autoregressive comparison without another search for the best lag. Since AR was added after the original evaluation, its comparison is labeled exploratory. I also expanded the original short-horizon forecast output by requesting a plot through 2026. That plot is kept separate from the evaluated forecasts and does not claim accuracy against unavailable 2026 actual prices.

**Project Results**

The results show that the models are close on the main one-session comparison. GARCH had a mean absolute error of 2.0169, AR had 2.0183, the last-price baseline had 2.0217, and ARIMA had 2.0241. Lower values are better, and these numbers are errors in the dataset's unverified price units rather than accuracy percentages. ARIMA had the lowest five-session MAE, while AR had the lowest twenty-session MAE in the expanded comparison. The original one-session difference between ARIMA and GARCH was statistically inconclusive, so I would not describe either model as a clear overall winner.

**Limitations**

There are still limitations. The historical trading calendar and the dataset's instrument details are unverified. Forecast horizons count the observations supplied in the CSV, and missing dates are not filled. The prediction intervals assume Gaussian innovations and do not include uncertainty in the fitted parameters, so they can miss extreme outcomes. Python 3.11 and 3.12 were not tested in the recorded review, and some dependency deprecation warnings remain. The initial simulation stability check used a simple ARIMA specification rather than the final selected order. The 2026 projection assumes a published NYSE holiday schedule, has not been validated at that long horizon, and should be read as a historical projection rather than a current forecast.

**Project Files**

The main implementation is in src/gold_forecast, the tests are in tests, and the default settings are in configs/default.toml. The architecture plan is saved in docs/plan.md, with implementation evidence in docs/implementation.md and review findings in docs/testing-review.md. Each output folder keeps the test forecasts, final forecasts, scores, configuration, and report separately so the results can be checked without mixing future projections with measured test accuracy.
