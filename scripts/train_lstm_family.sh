#!/bin/bash
# Train all LSTM family variants across seq_len 45, 20, 25
cd /Users/hdgehrke/Github_Cloning/nhl-data-rnn-transformers
LOG=/tmp/nhl_lstm_family.log
echo "=== LSTM family sweep started $(date) ===" >> $LOG

for cfg in lstm_small lstm_large lstm_deep; do
  for sl in 45 20 25; do
    echo "--- ${cfg} seq_len=${sl} $(date) ---" >> $LOG
    python -m src.train.run_experiment --config configs/${cfg}.yaml --seq-len ${sl} >> $LOG 2>&1
    echo "--- done ---" >> $LOG
  done
done

echo "=== LSTM family sweep done $(date) ===" >> $LOG
