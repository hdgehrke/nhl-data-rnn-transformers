#!/bin/bash
# Train all RNN family variants across seq_len 45, 20, 25
cd /Users/hdgehrke/Github_Cloning/nhl-data-rnn-transformers
LOG=/tmp/nhl_rnn_family.log
echo "=== RNN family sweep started $(date) ===" >> $LOG

for cfg in rnn_small rnn_large rnn_deep; do
  for sl in 45 20 25; do
    echo "--- ${cfg} seq_len=${sl} $(date) ---" >> $LOG
    python -m src.train.run_experiment --config configs/${cfg}.yaml --seq-len ${sl} >> $LOG 2>&1
    echo "--- done ---" >> $LOG
  done
done

echo "=== RNN family sweep done $(date) ===" >> $LOG
