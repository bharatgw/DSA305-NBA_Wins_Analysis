# What wins NBA games?

> Historical portfolio project from SMU DSA305. The repository preserves the report, data pipeline, and analysis artifacts from the original submission.

## Overview

This project analyses panel data for 30 NBA teams across 26 seasons. It studies team play style along three dimensions:

1. three-point versus two-point shooting;
2. teamwork versus star effects;
3. aggression versus defence.

It also examines how coaching experience, prior wins, and team familiarity relate to season wins, and estimates coach fixed effects. The original report attributes approximately 25% of variation in team wins to coach effects within its model and sample.

## Repository map

| Path | Purpose |
| --- | --- |
| [`data_extractNBAdata.ipynb`](./data_extractNBAdata.ipynb) | Historical extraction workflow for NBA and Basketball Reference data. |
| [`data_createFinaldf.ipynb`](./data_createFinaldf.ipynb) | Builds the panel-analysis dataset from files under `data/`. |
| [`analysis_PooledOLS-CoachEffects.ipynb`](./analysis_PooledOLS-CoachEffects.ipynb) | Pooled models and coach-effect analysis. |
| [`analysis_CRE_Model.py`](./analysis_CRE_Model.py) | Correlated random-effects analysis. |
| [`analysis_FE_RE_Models_w_Tests.py`](./analysis_FE_RE_Models_w_Tests.py) | Fixed- and random-effects models and specification tests. |
| [`outputTables/`](./outputTables) | Saved summary and regression tables. |
| [`[report] NBA_Wins_Analysis.pdf`](./%5Breport%5D%20NBA_Wins_Analysis.pdf) | Final project report. |
| [`DATA_SOURCES.md`](./DATA_SOURCES.md) | Source provenance, tracked-data map, and pipeline notes. |

## Historical workflow

1. Review or rerun [`data_extractNBAdata.ipynb`](./data_extractNBAdata.ipynb) only if the source sites still permit the requests.
2. Run [`data_createFinaldf.ipynb`](./data_createFinaldf.ipynb) to construct `data/finaldf.csv`.
3. Run the analysis notebooks and scripts against the prepared data.
4. Compare the saved outputs with the final report.

## Known reproducibility constraints

- Dependency versions are not pinned.
- [`analysis_FE_RE_Models_w_Tests.py`](./analysis_FE_RE_Models_w_Tests.py) reads `finaldf.csv` from the repository root, while the tracked file is `data/finaldf.csv`.
- The same script imports `Panel2RE_MLE`, which is not tracked in this repository.
- Source websites, page structures, and access terms may have changed since the extraction notebook was written.

Major Python dependencies include pandas, NumPy, Matplotlib, SciPy, `linearmodels`, requests, Beautiful Soup, Selenium, and tqdm.

## Data provenance and attribution

The repository contains data derived from NBA.com and Basketball Reference. Confirm current source terms before redistributing or refreshing those datasets. The project includes work by multiple contributors; consult the report and source-file headers for authorship details.

## Limitations

The estimates are conditional on the original variables, model specifications, sample, and data transformations. They are historical academic findings, not current sports forecasts or causal claims beyond the report's design.

## License and reuse

No open-source license has been applied. The project is shared for viewing as portfolio work. Group-authored material and third-party data remain subject to their respective rights.