# Independent implementation review

Reviewed October 4, 2026 against `docs/plan.md`.

The core implementation follows the planned observed-session splits, training-only
order selection, expanding-window evaluation, common-pair scoring, conditional
simulation, separate final refitting, and historical-as-of reporting. Tests cover
malformed inputs, future-data mutation, identical model histories, library
cross-checks, explicit fit failures, and seeded end-to-end reproducibility.

## Corrected finding

The completed-run auditor checked individual row alignment but did not require
the full scheduled model/origin/horizon grid. Missing or duplicated records could
therefore pass if the derived scores were regenerated consistently. It also did
not independently compare actual prices/returns with the source, fully validate
final forecast values, or compare all plotted forecast records with their source.

`scripts/verify_run.py` now checks these conditions, including finite positive
variance and nested positive price intervals. Seven regression cases exercise
valid input and missing, duplicated, altered-price, altered-return, infinite-
variance, and inverted-interval records.

## Verification evidence

- Existing suite before changes: 26 passed. Expanded suite: 33 passed.
- The documented locked installation succeeded in a separate temporary Python
  3.13 environment; the validation command reported the expected 1,760/501/405
  split and unchanged input checksum.
- Both saved complete runs passed the strengthened audit: 3,576 test rows,
  zero failed test rows, 60 final forecast rows, and 24 matching CSV artifacts
  at the declared tolerances. The primary report contained 10 valid embedded
  PNG images including the optional projection. The accuracy chart was also
  visually inspected.
- The presentation notebook executed successfully from a fresh kernel, writing
  `/private/tmp/gold-review-executed.ipynb` without modifying the source notebook.
- The roughly 42-minute full tuning/evaluation experiment was not rerun during
  this review; reproducibility checks compare the existing independent runs.

## Remaining limitations and recommendations

- Python 3.11 and 3.12 compatibility remains untested in this review. Add a CI
  matrix before claiming verified support across all advertised versions.
- NumPy compatibility deprecation warnings originate in the pinned scientific
  stack. Recheck these when updating the lockfile; current checks pass.
- The presentation notebook unconditionally includes the calendar-2026 extension,
  which supports only the supplied August 14, 2025 cutoff. A custom dataset with
  a different cutoff requires skipping that optional section; document or guard
  this before broadening dataset support.
- Instrument provenance and the historical exchange calendar remain unverified,
  as disclosed. The 2026 extension is not a validated long-horizon forecast.
- The preselection simulation stability audit uses ARIMA (0,0), not every
  candidate or the selected (3,1). Audit the selected specification's quantiles
  on training-only origins before making stronger precision claims.
