# NHL/MLB Game Prediction with RNNs and Transformers

An end-to-end machine learning pipeline that predicts game **run/goal differential** using per-team rolling game sequences. Models are trained on 14 seasons of NHL and MLB data (2011–2024) and evaluated against several baselines. Cross-sport transfer experiments test whether patterns learned from hockey generalize to basketball (NBA).

## Overview

Each game is represented as a pair of sequences — one for each team — where each element in the sequence is a vector of stats from one of that team's recent games. A shared encoder (RNN, LSTM, GRU, or Transformer) reads each team's sequence independently and produces a fixed-size embedding. The two embeddings are then concatenated and passed through an MLP head to predict the final goal differential.

```
Team A: [game_1, ..., game_N] → Encoder → embed_A
Team B: [game_1, ..., game_N] → Encoder → embed_B
[embed_A, embed_B, context] → MLP → predicted goal differential
```

Sharing encoder weights between the two teams enforces symmetry and halves the parameter count.

## NHL Results

### Architecture sweep at seq_len=45

19 model configurations (4 variants per RNN/LSTM/GRU family + 7 transformer variants) trained on NHL data at seq_len=45. All beat all three baselines.

#### RNN family

| Variant | Parameters | hidden_dim | layers | MAE (sl=20) | MAE (sl=25) | MAE (sl=45) | Win Acc (sl=45) |
|---|---|---|---|---|---|---|---|
| base | 95K | 128 | 2 | 2.208 | 2.206 | 2.199 | 57.7% |
| small | 29K | 64 | 1 | 2.201 | 2.200 | **2.187** | **58.2%** |
| large | 362K | 256 | 2 | 2.214 | 2.203 | 2.193 | 57.8% |
| deep | 193K | 128 | 4 | 2.207 | 2.207 | 2.235 | 55.7% |

#### LSTM family

| Variant | Parameters | hidden_dim | layers | MAE (sl=20) | MAE (sl=25) | MAE (sl=45) | Win Acc (sl=45) |
|---|---|---|---|---|---|---|---|
| **small** | **36K** | 64 | 1 | **2.199** | **2.191** | **2.164** | **59.4%** |
| base | 255K | 128 | 2 | 2.210 | 2.201 | 2.166 | 58.8% |
| deep | 509K | 128 | 4 | 2.205 | 2.201 | 2.179 | 58.7% |
| large | 1.01M | 256 | 2 | 2.206 | 2.202 | 2.183 | 58.4% |

#### GRU family

| Variant | Parameters | hidden_dim | layers | MAE (sl=20) | MAE (sl=25) | MAE (sl=45) | Win Acc (sl=45) |
|---|---|---|---|---|---|---|---|
| **deep** | **402K** | 128 | 4 | 2.199 | 2.192 | **2.159** | 57.7% |
| base | 201K | 128 | 2 | 2.210 | 2.189 | 2.168 | 58.2% |
| small | 50K | 64 | 1 | 2.198 | 2.189 | 2.168 | **58.3%** |
| large | 802K | 256 | 2 | 2.193 | 2.189 | 2.186 | 58.6% |

#### Transformer family

| Variant | Parameters | d_model | layers | MAE (sl=20) | MAE (sl=25) | MAE (sl=45) | Win Acc (sl=45) |
|---|---|---|---|---|---|---|---|
| **small** | **311K** | 128 | 2 | **2.195** | **2.184** | **2.149** | 59.1% |
| medium_shallow | 1.14M | 256 | 2 | 2.199 | 2.192 | 2.149 | 59.4% |
| large | 9.08M | 512 | 4 | 2.200 | 2.207 | 2.151 | **59.4%** |
| small_deep | 622K | 128 | 4 | 2.195 | 2.189 | 2.158 | 59.3% |
| deep | 2.28M | 256 | 6 | 2.191 | 2.185 | 2.156 | 58.9% |
| tiny | 79K | 64 | 2 | 2.202 | 2.184 | 2.160 | 58.9% |
| medium | 2.28M | 256 | 4 | 2.199 | 2.182 | 2.172 | **59.6%** |

| *Baseline: home +0.3* | — | — | — | — | — | *2.258* | *55.2%* |
| *Baseline: team avg* | — | — | — | — | — | *2.272* | *54.2%* |
| *Baseline: always 0* | — | — | — | — | — | *2.289* | *0.0%* |

### Effect of sequence length — original 5 models (Test MAE)

| Model | 10 | 20 | 25 | 30 | 40 | 45 |
|---|---|---|---|---|---|---|
| Transformer Small | 2.220 | 2.195 | 2.184 | 2.176 | 2.163 | **2.149** |
| Transformer Medium | 2.221 | 2.199 | 2.182 | 2.180 | **2.157** | 2.172 |
| GRU | 2.224 | 2.210 | 2.189 | 2.179 | 2.174 | 2.168 |
| LSTM | **2.214** | 2.210 | 2.201 | 2.196 | 2.175 | 2.166 |
| RNN | 2.231 | 2.208 | 2.206 | 2.209 | 2.230 | 2.199 |

**Key findings across the architecture sweep:**
- All 19 configurations beat all three baselines at every sequence length tested.
- **Transformers dominate at seq_len=45** — the top 4 MAE scores all belong to transformer variants. The Transformer Small (2.149) and Medium Shallow (2.149) tie for best MAE; Medium has the best win accuracy (59.6%).
- **Bigger ≠ better for transformers**: The 9M-parameter Large barely edges the 311K Small (2.151 vs 2.149 MAE), suggesting the task saturates well below maximum capacity.
- **LSTM Small surprises**: The single-layer 36K LSTM (2.164 MAE, 59.4% win acc) matches the full 255K base LSTM — strong inductive bias for sequential data at this scale.
- **RNN degrades with depth**: rnn_deep (4 layers) is the *worst* RNN variant at seq_len=45 (2.235 MAE, 55.7% win acc), consistent with vanishing gradient issues.
- **GRU benefits from depth**: gru_deep (4 layers, 2.159 MAE) is the best GRU variant, outperforming gru_large despite fewer parameters.
- At shorter seq_lens (20–25), differences between variants compress significantly — architecture matters most when longer context is available.

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

## MLB Results

Five model architectures trained natively on MLB data (2011–2024). **Feature mapping**: runs→goals, hits→shots, errors→giveaways, extra innings→OT flag. NHL-specific features (Corsi, xGoals, power play, physical hits) zero-filled.

**MLB data**: 14 seasons (2011–2024), train 2011–2021, val 2022, test 2023–2024 (~4,860 team-game rows/season; 4,859 test games).

| Model | Parameters | Test MAE | Test RMSE | Win Acc |
|---|---|---|---|---|
| **GRU** | 201K | **3.455** | **4.408** | **55.4%** |
| Transformer Small | 311K | 3.457 | 4.414 | 54.6% |
| Transformer Medium | 2.28M | 3.457 | 4.416 | 55.0% |
| LSTM | 255K | 3.468 | 4.415 | 54.9% |
| RNN | 95K | 3.469 | 4.431 | 54.0% |
| *Baseline: team avg* | — | *3.523* | *4.459* | *52.2%* |
| *Baseline: home +0.3* | — | *3.526* | *4.480* | *52.1%* |
| *Baseline: always 0* | — | *3.538* | *4.467* | *0.0%* |

**Key findings**:
- All models beat all baselines, but margins are tighter than NHL (best model beats strongest baseline by 0.068 MAE vs 0.109 for NHL). Baseball run differential is inherently harder to predict — high variance, small home advantage (~0.1–0.3 runs), and more randomness per game than hockey.
- Models are tightly clustered within 0.014 MAE, suggesting the available features (runs, hits, errors, home/away, rest) are close to the predictability ceiling without pitching or advanced stats.
- The Transformer Medium overfit noticeably (train MSE → 18.0 while val plateaued at 18.6+), indicating the larger architecture has excess capacity relative to the signal in the zero-filled feature space.
- Win accuracy of 54–55% compares modestly to NHL's 58–60%, consistent with the higher per-game randomness in baseball.

To reproduce:
```bash
# Download and process MLB data
python -m src.fetch.mlb_client --start 2011 --end 2024
python -m src.features.mlb_tokenizer --start 2011 --end 2024

# Train all models
python -m src.train.run_experiment --config configs/mlb_rnn.yaml
python -m src.train.run_experiment --config configs/mlb_lstm.yaml
python -m src.train.run_experiment --config configs/mlb_gru.yaml
python -m src.train.run_experiment --config configs/mlb_transformer_small.yaml
python -m src.train.run_experiment --config configs/mlb_transformer_medium.yaml
```

## Features

Each game token is a 28-dimensional vector including:
- **Outcome**: goals/runs for/against, differential, win/loss, OT/extra-innings flag
- **Volume**: shots/hits for/against, hits, blocks, giveaways, takeaways, PIM
- **Efficiency**: power-play %, penalty-kill %, shooting %, save %
- **Context**: home/away flag, rest days since last game, back-to-back flag, game number in season

Sport-specific fields unavailable for a given sport are zero-filled (e.g. Corsi/xGoals/PP% for MLB/NBA).

## Project Structure

```
nhl-data-rnn-transformers/
├── data/
│   ├── raw/              # Raw JSON from NHL/MLB/NBA APIs (gitignored)
│   └── processed/        # Parquet files per season (gitignored)
├── src/
│   ├── fetch/            # NHL, MLB, NBA API clients and download CLIs
│   ├── features/         # GameToken schema, NHL/MLB/NBA tokenizers, sequence
│   │                     #   builder, normalization, PyTorch Dataset
│   ├── models/           # RNN, LSTM, GRU, Transformer, shared MLP head,
│   │                     #   model registry
│   ├── train/            # Training loop with MLflow logging, CLI entry point
│   └── eval/             # Metrics, baselines, calibration plots, NBA transfer
├── configs/              # YAML hyperparameter files per model/sport variant
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

### NHL

```bash
# 1. Download raw game data (api.nhle.com)
python -m src.fetch.download --start 2011 --end 2024

# 2. Process into Parquet (data/processed/games_XXXXXXXX.parquet)
python -m src.features.tokenizer

# 3. Train models
python -m src.train.run_experiment --config configs/lstm_base.yaml
python -m src.train.run_experiment --config configs/gru_base.yaml
python -m src.train.run_experiment --config configs/rnn_base.yaml
python -m src.train.run_experiment --config configs/transformer_small.yaml
python -m src.train.run_experiment --config configs/transformer_medium.yaml
```

The 2004–05 season (cancelled lockout) is skipped automatically. You can override sequence length: `--seq-len 20`.

### MLB

```bash
# 1. Download raw game data (statsapi.mlb.com — one call per season)
python -m src.fetch.mlb_client --start 2011 --end 2024

# 2. Process into Parquet (data/processed/mlb_games_{year}.parquet)
python -m src.features.mlb_tokenizer --start 2011 --end 2024

# 3. Train models
python -m src.train.run_experiment --config configs/mlb_rnn.yaml
python -m src.train.run_experiment --config configs/mlb_lstm.yaml
python -m src.train.run_experiment --config configs/mlb_gru.yaml
python -m src.train.run_experiment --config configs/mlb_transformer_small.yaml
python -m src.train.run_experiment --config configs/mlb_transformer_medium.yaml
```

The 2020 season was shortened (60 games); it is included as-is.

Training uses AdamW with a ReduceLROnPlateau scheduler and early stopping (patience=10). On an M2 Mac, RNN/LSTM/GRU train in under 2 minutes per sport; Transformers take 5–20 minutes.

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
