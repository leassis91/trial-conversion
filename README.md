# Trial Conversion Model

Predicts whether a free-trial user will convert to a paid subscription, based on
their first three days of product activity.

Built as part of the **Futureproof MLOps cohort**. The goal is not just a model
that scores well, but a project structured the way production code is: logic in
an installable package, reproducible commands, versioned data extracts, and
metrics saved per run.

## What it does

| Stage | Command | Output |
|---|---|---|
| Extract | `make fetch` | `data/01_raw/trials_raw.csv` (~1,500 rows) |
| Train | `make train` | `models/xgb-<run_id>.pkl` and `models/metrics-<run_id>.json` |

Each run is tagged with a UTC timestamp (`run_id`), so a model file and its
metrics file always point to the same experiment.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
cp .env.example .env   # then fill in the database credentials
```

Environment variables:

```dotenv
DB_HOST=
DB_PORT=5432
DB_NAME=
DB_USER=
DB_PASSWORD=
CURRENT_TABLE=
```

## Usage

```bash
make all     # list available commands
make fetch    # pull the training extract from Postgres
make train    # build features, train XGBoost, save model and metrics
```

Comparing runs:

```python
from config.config import MODELS_DIR
from trial_conversion_model.evaluate import load_metrics

load_metrics(MODELS_DIR).sort_values("metrics.roc_auc", ascending=False)
```

## Layout

```
.
├── config/                        # project config: paths, parameters
│   └── config.py
├── data/                          # local only, not versioned
│   └── 01_raw/
├── models/                        # local only, not versioned
├── notebooks/                     # thin shell, imports from the package
├── scripts/                       # entry points
|   ├── evaluate.py                # evaluates latest model
│   ├── fetch_data.py
│   └── train.py
└── src/trial_conversion_model/    # the library
    ├── data.py                    # database extraction
    ├── features.py                # feature engineering, train/test split
    ├── train.py                   # model training and persistence
    
    └── predict.py                 # inference
```

Two conventions worth naming, since both were deliberate:

**The package takes paths, it never guesses them.** `load_data(output_dir)` and
`save_model(model, filename, output_dir)` receive a `Path`. Only `config/config.py`
resolves the project root, anchored on `__file__` rather than the working
directory, so the commands behave the same from anywhere.

**Training reads a saved extract, not the live table.** Querying
`ml.trial_snapshot_latest` directly would be one step fewer, but `_latest` is a
moving target: two runs would no longer be comparable. With a frozen snapshot, a
drop in metrics can be traced to the code, the parameters, or the data, instead
of all three at once. Freshness matters at inference time; at training time what
matters is that the input holds still.

## Notes

`data/` and `models/` are gitignored. Run `make fetch` after cloning.

Metrics are stored as one JSON file per run. No experiment tracker yet, which is
a deliberate simplification: the `run_id` in each payload is doing informally
what a tool like MLflow or DVC would do properly.