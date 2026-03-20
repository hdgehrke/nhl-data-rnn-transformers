#!/bin/bash
# Run all NHL architecture sweep training jobs sequentially.
# Avoids MLflow SQLite contention from parallel writes.
# Already completed (skip): rnn_deep/45, rnn_large/45, gru_small/45, lstm_small/45
cd /Users/hdgehrke/Github_Cloning/nhl-data-rnn-transformers
LOG=/tmp/nhl_sweep_sequential.log
echo "=== Sequential sweep started $(date) ===" | tee -a $LOG

run() {
  cfg=$1; sl=$2
  echo "--- ${cfg} seq_len=${sl} $(date) ---" | tee -a $LOG
  python -m src.train.run_experiment --config configs/${cfg}.yaml --seq-len ${sl} 2>&1 | tee -a $LOG
  echo "" >> $LOG
}

# RNN family
run rnn_small 45
run rnn_small 20
run rnn_small 25
run rnn_large 20
run rnn_large 25
run rnn_deep  20
run rnn_deep  25

# LSTM family
run lstm_small 20
run lstm_small 25
run lstm_large 45
run lstm_large 20
run lstm_large 25
run lstm_deep  45
run lstm_deep  20
run lstm_deep  25

# GRU family
run gru_small 20
run gru_small 25
run gru_large 45
run gru_large 20
run gru_large 25
run gru_deep  45
run gru_deep  20
run gru_deep  25

# Transformer family (new variants)
run transformer_tiny           45
run transformer_tiny           20
run transformer_tiny           25
run transformer_small_deep     45
run transformer_small_deep     20
run transformer_small_deep     25
run transformer_medium_shallow 45
run transformer_medium_shallow 20
run transformer_medium_shallow 25
run transformer_deep           45
run transformer_deep           20
run transformer_deep           25
run transformer_large          45
run transformer_large          20
run transformer_large          25

echo "=== Sequential sweep done $(date) ===" | tee -a $LOG
