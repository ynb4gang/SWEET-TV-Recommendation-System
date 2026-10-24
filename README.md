# SWEET.TV Recommendation System

HSE LLM4Rec Midterm Quiz — TV programme recommendation for 1,260 users.

**Designated Kaggle submission:** `outputs/submission_objective_v5.csv`  
**Returned Kaggle MAP@5:** Private **0.43744**, Public **0.45154** (24 October 2026). The authoritative evidence is the Kaggle submission-page screenshot in the separate PDF report.

## Implemented algorithms

- Global popularity baseline
- Personal frequency and recency baselines
- Content-based filtering (CBF) using programme metadata
- Personalized PageRank (PPR) on sparse co-view relationships
- Collaborative item-neighbour signals and LightGBM LambdaRank, with context/exposure features and a Top-5 frequency-label proxy

CBF and PPR are independently implemented; popularity is the mandatory reference. The final hybrid uses bounded eligible candidate retrieval and supervised ranking.

## Project layout

- `src/`: viewing-to-schedule matching, metadata modelling, ranking and CSV validation
- `scripts/`: interaction creation, baselines, temporal validation and phases 5–9
- `tests/`: deterministic unit/integrity tests
- `evidence/`: machine-readable recorded measurements from completed experiments
- `outputs/submission_objective_v5.csv`: the designated, Kaggle-verified CSV

## Data and evaluation

The raw SWEET.TV files are **not** included. After accepting the Kaggle competition rules, download the dataset to `data/`:

- `dataset11-30.csv`
- `export_arh_11-20-final.csv`
- `export_arh_21-30-final.csv`
- `export_arh_31-42-final.csv`
- `submission.csv`

The pipeline matches historical viewing sessions to programme broadcasts using channel and interval overlap and excludes unidentified programme IDs. Local validation is chronological, and the metric is MAP@5. Experiments using the all-positive target and those using the Top-5 frequency proxy are **not directly numerically comparable**.

## Setup and checks

Use Python with compatible versions of `numpy`, `scipy`, `polars`, `duckdb`, `scikit-learn`, `lightgbm`, and `pytest`. A local environment is recommended; source files do not include the large cached intermediate tables.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install numpy scipy polars duckdb scikit-learn lightgbm pytest pandas
python -m pytest tests/ -q
python scripts/validate_submission.py outputs/submission_objective_v5.csv
```

Experiment scripts include `scripts/evaluate_baselines.py`, `scripts/evaluate_cbf.py`, `scripts/optimize_and_submit.py`, `scripts/phase06_optimize.py`, `scripts/phase08_align.py`, and `scripts/phase09_optimize.py`. Rebuilding full experiments requires the raw dataset, intermediate artifacts, and compute time; the existing evidence should not be construed as an independent re-execution.

## Provenance

The development history is preserved in the **original, unmodified OpenCode session export**, submitted separately according to the course instructions. Unsuccessful experiments are also documented. No live A/B test or business revenue lift is claimed.
