# Data sources and pipeline notes

## Provenance

The project assembled historical team, player, coaching, and award information from NBA.com and Basketball Reference. The tracked CSV files are derived research artifacts, not official redistributions by those providers.

Before refreshing or redistributing the data, review the current source terms, robots policies, request limits, definitions, and attribution requirements.

## Tracked data groups

| Files | Role |
| --- | --- |
| `data/team*Stats.csv` | Team-level traditional, advanced, opponent, scoring, defence, miscellaneous, and four-factor statistics. |
| `data/playerTradStats.csv` | Player-level traditional statistics used in feature construction. |
| `data/coachdf.csv` | Coaching history and team associations used by the models. |
| `data/coach_of_the_year.csv` and `data/awards.csv` | Coaching and player award information. |
| `data/abbreviations.csv` | Identifier mapping used when combining sources. |
| `data/finaldf.csv` | Prepared team-season panel used by the main analyses. |

## Pipeline order

1. `data_extractNBAdata.ipynb` collects source data.
2. `data_createFinaldf.ipynb` combines and transforms the tracked CSV files.
3. The analysis notebook and scripts consume `data/finaldf.csv`.
4. `outputTables/` stores selected reporting outputs.

## Reproducibility limits

Source endpoints and HTML structures may have changed. The extraction notebook should not be rerun without confirming current permission and request-rate expectations. The repository also lacks pinned dependency versions, and one analysis script references an untracked module as documented in the main README.
