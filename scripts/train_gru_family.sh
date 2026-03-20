#!/bin/bash
# Train all GRU family variants across seq_len 45, 20, 25
cd /Users/hdgehrke/Github_Cloning/nhl-data-rnn-transformers
LOG=/tmp/nhl_gru_family.log
echo "=== GRU family sweep started $(date) ===" >> $LOG

for cfg in gru_small gru_large gru_deep; do
  for sl in 45 20 25; do
    echo "--- ${cfg} seq_len=${sl} $(date) ---" >> $LOG
    python -m src.train.run_experiment --config configs/${cfg}.yaml --seq-len ${sl} >> $LOG 2>&1
    echo "--- done ---" >> $LOG
  done
done

echo "=== GRU family sweep done $(date) ===" >> $LOG
