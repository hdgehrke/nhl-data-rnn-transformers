# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

End-to-end ML pipeline to predict NHL game **goal differential** (home - away) using per-team rolling game sequences. Models: vanilla RNN, LSTM, GRU, Transformer (small + medium). Experiments tracked with MLflow locally. Architecture is extensible to MLB/NBA.

## Development Setup

```bash
pip install -r requirements.txt
```

Linting:
```bash
ruff check .
```

Tests:
```bash
pytest tests/
```

## Data Pipeline

**Step 1 — Download raw game data:**
```bash
python -m src.fetch.download --start 2011 --end 2024
# outputs: data/raw/games_XXXXXXXX.json (one per season)
```

**Step 2 — Process into Parquet:**
```bash
python -m src.features.tokenizer
# outputs: data/processed/games_XXXXXXXX.parquet
```

## Training

Train any model variant from a YAML config:
```bash
python -m src.train.run_experiment --config configs/lstm_base.yaml
python -m src.train.run_experiment --config configs/transformer_small.yaml
```

View experiment results:
```bash
mlflow ui
```

## Project Structure

```
src/
  fetch/        — NHL API client (nhl_client.py), download CLI (download.py)
  features/     — GameToken schema, tokenizer, sequence builder, normalization, Dataset
  models/       — rnn.py, lstm.py, gru.py, transformer.py, base.py, registry.py
  train/        — training loop + MLflow (trainer.py), CLI entry point (run_experiment.py)
  eval/         — metrics.py (MAE/RMSE/win-acc, baselines), calibration.py (plots)
configs/        — YAML hyperparameter files per model variant
tests/          — unit tests (28 passing)
data/raw/       — JSON from NHL API (gitignored)
data/processed/ — Parquet per season (gitignored)
```

## Key Design Decisions

- **Chronological splits**: train 2011–2022, val 2022–23, test 2023–24. No shuffling across time.
- **Per-train-split z-score normalization**: scaler fit on train sequences only, applied to val/test.
- **Shared encoder weights**: both team A and team B use the same encoder (parameter-efficient, enforces symmetry).
- **Padding mask**: True = padded/invalid token. Early-season games zero-padded at the front.
- **MPS support**: `get_device()` in trainer.py checks for Apple MPS before CUDA before CPU.
- **Feature dimension**: 28 features per game token (see `src/features/schema.py:FEATURE_NAMES`).
- **Skipped seasons**: 2004–05 (cancelled). 2012–13 and 2020–21 are short seasons but included.

## Verification Checklist

- After `download.py`: JSON files in `data/raw/` with expected record counts
- After `tokenizer.py`: Parquet files in `data/processed/`; verify `len(df)` ≈ 2× games per season
- After training: check MLflow UI for train/val loss curves; model should beat `baseline_zero` MAE
- No data leakage: val/test sequences only use game history from before the split date
