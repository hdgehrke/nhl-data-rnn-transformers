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

### seq_len=45 (best configuration)

| Model | Parameters | Test MAE | Test RMSE | Win Acc |
|---|---|---|---|---|
| **Transformer Small** | 311K | **2.149** | **2.564** | 59.1% |
| Transformer Medium | 2.28M | 2.172 | 2.570 | **59.6%** |
| LSTM | 255K | 2.166 | 2.576 | 58.8% |
| GRU | 201K | 2.168 | 2.580 | 58.2% |
| RNN | 95K | 2.199 | 2.599 | 57.7% |
| *Baseline: home +0.3* | — | *2.258* | *2.634* | *55.2%* |
| *Baseline: team avg* | — | *2.272* | *2.640* | *54.2%* |
| *Baseline: always 0* | — | *2.289* | *2.648* | *0.0%* |

### Effect of sequence length (Test MAE)

| Model | 10 | 20 | 25 | 30 | 40 | 45 |
|---|---|---|---|---|---|---|
| Transformer Small | 2.220 | 2.195 | 2.184 | 2.176 | 2.163 | **2.149** |
| Transformer Medium | 2.221 | 2.199 | 2.182 | 2.180 | **2.157** | 2.172 |
| GRU | 2.224 | 2.210 | 2.189 | 2.179 | 2.174 | 2.168 |
| LSTM | **2.214** | 2.210 | 2.201 | 2.196 | 2.175 | 2.166 |
| RNN | 2.231 | 2.208 | 2.206 | 2.209 | 2.230 | 2.199 |

All trained models beat all three baselines at every sequence length. The Transformer Small continues to improve through seq_len=45 (2.149 MAE, 59.1% win acc) with no sign of saturation — attention benefits most from longer context. The LSTM rebounds strongly at longer sequences, nearly matching the GRU by seq_len=45. The RNN is unreliable beyond seq_len=30, oscillating rather than improving consistently. The best model (Transformer Small, seq_len=45) beats the strongest baseline (home +0.3) by 0.109 MAE, compared to only 0.044 at seq_len=10.

## NBA Cross-Sport Transfer

All five NHL model families were trained on the full dataset (2011–2024, no test holdout) and then evaluated on NBA game data to test whether momentum/form patterns learned from hockey generalize to basketball.

**Feature mapping**: NBA box score stats are mapped onto the shared 28-feature GameToken schema. Shared features (points for/against, shots/FGA, blocks, steals/takeaways, turnovers/giveaways, home/away, rest days) are populated; NHL-specific features (hits, Corsi, Fenwick, xGoals, power-play stats) are zero-filled.

**NBA data**: 10 seasons (2015–2025), split into train 2015–2021, val 2021-22, test 2022–2024 (3,685 games).

### Zero-shot results (NHL weights, no NBA training)

| Model | MAE | RMSE | Win Acc |
|---|---|---|---|
| Transformer Medium | 11.93 | 14.97 | 64.2% |
| Transformer Small | 11.94 | 14.97 | 64.0% |
| RNN | 11.95 | 14.97 | 64.2% |
| GRU | 11.94 | 14.96 | 62.9% |
| LSTM | 11.99 | 15.01 | 63.4% |
| *Baseline: home +3.0* | *11.96* | *15.10* | *55.6%* |
| *Baseline: always 0* | *12.16* | *15.22* | *0.0%* |

### Fine-tuned results (NHL pretrain → NBA fine-tune)

| Model | MAE | RMSE | Win Acc |
|---|---|---|---|
| **Transformer Small** | **10.79** | **13.80** | **64.6%** |
| GRU | 10.79 | 13.81 | **64.9%** |
| Transformer Medium | 10.80 | 13.82 | 64.5% |
| LSTM | 10.88 | 13.89 | 64.3% |
| RNN | 10.94 | 13.97 | 63.6% |
| *Baseline: home +3.0* | *11.96* | *15.10* | *55.6%* |
| *Baseline: always 0* | *12.16* | *15.22* | *0.0%* |

**Key findings**:
- All five NHL zero-shot models beat both baselines on win direction accuracy (**62.9–64.2% vs 55.6%**), confirming that sequential momentum and home/away form patterns learned from hockey genuinely transfer to basketball — regardless of architecture.
- Fine-tuning on NBA training data reduces MAE by ~1.1 points across all models (adapting to the larger NBA point scale) and further improves win accuracy.
- The Transformers and GRU lead fine-tuned win accuracy (64.5–64.9%), while the RNN lags slightly (63.6%), consistent with its weaker long-context representation.
- All fine-tuned models comfortably outperform both baselines on all three metrics.

To reproduce:
```bash
# Download and process NBA data
python -m src.fetch.nba_client --start 2015 --end 2024
python -m src.features.nba_tokenizer

# Train full-data NHL models (no test holdout)
python -m src.train.run_experiment --config configs/transformer_small_fulldata.yaml
python -m src.train.run_experiment --config configs/transformer_medium_fulldata.yaml
python -m src.train.run_experiment --config configs/lstm_fulldata.yaml
python -m src.train.run_experiment --config configs/gru_fulldata.yaml
python -m src.train.run_experiment --config configs/rnn_fulldata.yaml

# Run transfer eval (zero-shot + fine-tune) per model
python -m src.eval.nba_transfer --nhl-run-id <mlflow_run_id>
```

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
│   ├── fetch/            # NHL + NBA API clients and download CLIs
│   ├── features/         # GameToken schema, NHL/NBA tokenizers, sequence
│   │                     #   builder, normalization, PyTorch Dataset
│   ├── models/           # RNN, LSTM, GRU, Transformer, shared MLP head,
│   │                     #   model registry
│   ├── train/            # Training loop with MLflow logging, CLI entry point
│   └── eval/             # Metrics, baselines, calibration plots, NBA transfer
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
