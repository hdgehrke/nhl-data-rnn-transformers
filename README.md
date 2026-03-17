# NHL Game Prediction with RNNs and Transformers

An end-to-end machine learning pipeline that predicts NHL game **goal differential** (home team goals − away team goals) using per-team rolling game sequences. Models are trained on 14 seasons of NHL data (2011–2025) and evaluated against several baselines.

## Overview

Each game is represented as a pair of sequences — one for each team — where each element in the sequence is a vector of stats from one of that team's recent games. A shared encoder (RNN, LSTM, GRU, or Transformer) reads each team's sequence independently and produces a fixed-size embedding. The two embeddings are then concatenated and passed through an MLP head to predict the final goal differential.

```
Team A: [game_1, ..., game_N] → Encoder → embed_A
Team B: [game_1, ..., game_N] → Encoder → embed_B
[embed_A, embed_B, context] → MLP → predicted goal differential
```

Sharing encoder weights between the two teams enforces symmetry and halves the parameter count.

## Results

### seq_len=30 (best configuration)

| Model | Parameters | Test MAE | Test RMSE | Win Acc |
|---|---|---|---|---|
| **Transformer Small** | 311K | **2.176** | 2.582 | **58.2%** |
| Transformer Medium | 2.28M | 2.180 | **2.580** | 57.9% |
| GRU | 201K | 2.179 | 2.592 | **58.2%** |
| LSTM | 255K | 2.196 | 2.596 | 56.4% |
| RNN | 95K | 2.209 | 2.601 | 57.9% |
| *Baseline: home +0.3* | — | *2.258* | *2.634* | *55.2%* |
| *Baseline: team avg* | — | *2.272* | *2.640* | *54.2%* |
| *Baseline: always 0* | — | *2.289* | *2.648* | *0.0%* |

### Effect of sequence length (Test MAE)

| Model | seq_len=10 | seq_len=20 | seq_len=25 | seq_len=30 |
|---|---|---|---|---|
| Transformer Small | 2.220 | **2.195** | 2.184 | **2.176** |
| Transformer Medium | 2.221 | 2.199 | **2.182** | 2.180 |
| GRU | 2.224 | 2.210 | 2.189 | **2.179** |
| LSTM | **2.214** | 2.210 | 2.201 | 2.196 |
| RNN | 2.231 | 2.208 | 2.206 | 2.209 |

All trained models beat all three baselines at every sequence length. Longer sequences consistently improve the Transformers and GRU — the Transformer Small leads at seq_len=30 on MAE, tied with the GRU on win accuracy at 58.2%. The LSTM and RNN show diminishing returns beyond seq_len=25, as recurrent models struggle to retain signal over longer sequences compared to attention-based models.

## Features

Each game token is a 28-dimensional vector including:
- **Outcome**: goals for/against, goal differential, win/loss, OT flag
- **Volume**: shots for/against, hits, blocks, giveaways, takeaways, PIM
- **Efficiency**: power-play %, penalty-kill %, shooting %, save %
- **Context**: home/away flag, rest days since last game, back-to-back flag, game number in season

## Project Structure

```
nhl-data-rnn-transformers/
├── data/
│   ├── raw/              # Raw JSON from NHL API (gitignored)
│   └── processed/        # Parquet files per season (gitignored)
├── src/
│   ├── fetch/            # NHL API client and download CLI
│   ├── features/         # GameToken schema, tokenizer, sequence builder,
│   │                     #   normalization, PyTorch Dataset
│   ├── models/           # RNN, LSTM, GRU, Transformer, shared MLP head,
│   │                     #   model registry
│   ├── train/            # Training loop with MLflow logging, CLI entry point
│   └── eval/             # Metrics, baselines, calibration plots
├── configs/              # YAML hyperparameter files per model variant
├── notebooks/            # Calibration plots
├── tests/                # 28 unit tests
├── requirements.txt
└── CLAUDE.md
```

## Setup

```bash
git clone https://github.com/hdgehrke/nhl-data-rnn-transformers.git
cd nhl-data-rnn-transformers
pip install -r requirements.txt
```

Requires Python 3.11+. Training uses Apple MPS automatically on M-series Macs, falling back to CUDA then CPU.

## Running the Pipeline

### 1. Download raw game data

Fetches regular-season game records from `api.nhle.com` for each season and saves them as JSON to `data/raw/`.

```bash
python -m src.fetch.download --start 2011 --end 2024
```

The 2004–05 season (cancelled lockout) is skipped automatically. Short seasons (2012–13, 2020–21) are included as-is.

### 2. Process into Parquet

Parses raw JSON into per-team game tokens, computes rest days, back-to-back flags, and game numbers, and saves one Parquet file per season to `data/processed/`.

```bash
python -m src.features.tokenizer
```

Each season produces approximately 2× the number of games (one row per team per game).

### 3. Train a model

Train any model variant using its YAML config. Experiments are logged to MLflow automatically.

```bash
python -m src.train.run_experiment --config configs/lstm_base.yaml
python -m src.train.run_experiment --config configs/gru_base.yaml
python -m src.train.run_experiment --config configs/rnn_base.yaml
python -m src.train.run_experiment --config configs/transformer_small.yaml
python -m src.train.run_experiment --config configs/transformer_medium.yaml
```

You can override the sequence length at the command line:

```bash
python -m src.train.run_experiment --config configs/lstm_base.yaml --seq-len 20
```

Training uses AdamW with a ReduceLROnPlateau scheduler and early stopping (patience=10). On an M2 Mac, RNN/LSTM/GRU train in under a minute; Transformers take 2–5 minutes.

### 4. Run baselines

Computes and logs three baselines to MLflow: always predict 0, always predict home +0.3, and season-average team differential.

```bash
python -m src.eval.run_baselines
```

### 5. View experiment results

```bash
mlflow ui
```

Then open http://127.0.0.1:5000. All model variants and baselines are logged as separate MLflow experiments.

### 6. Run tests

```bash
pytest tests/
```

28 tests covering the feature schema, sequence builder (including data leakage checks), normalization, and all model architectures.

### 7. Lint

```bash
ruff check .
```

## Data Splits

Training uses a strict chronological split with no shuffling across time boundaries to prevent data leakage:

| Split | Seasons | Games |
|---|---|---|
| Train | 2011–12 through 2021–22 | ~12,674 |
| Val | 2022–23 | ~1,312 |
| Test | 2023–24 | ~2,624 |

Features are z-score normalized using statistics computed on the training set only.

## Design Notes

- **Shared encoder weights**: Both teams pass through the same encoder, enforcing symmetry and halving parameters.
- **Zero-padding**: Teams with fewer than `seq_len` games in the season are front-padded with zeros; a boolean mask prevents the model from attending to padding.
- **MPS support**: Device selection automatically prefers Apple MPS → CUDA → CPU.
- **Extensibility**: The `GameToken` dataclass has a `sport` field and an `extras` dict for sport-specific data, making it straightforward to add MLB or NBA fetchers.
