#!/bin/bash
# Train all new transformer variants across seq_len 45, 20, 25
# Also re-runs original small/medium at seq_len 20/25 if not already done
cd /Users/hdgehrke/Github_Cloning/nhl-data-rnn-transformers
LOG=/tmp/nhl_transformer_family.log
echo "=== Transformer family sweep started $(date) ===" >> $LOG

for cfg in transformer_tiny transformer_small_deep transformer_medium_shallow transformer_deep transformer_large; do
  for sl in 45 20 25; do
    echo "--- ${cfg} seq_len=${sl} $(date) ---" >> $LOG
    python -m src.train.run_experiment --config configs/${cfg}.yaml --seq-len ${sl} >> $LOG 2>&1
    echo "--- done ---" >> $LOG
  done
done

echo "=== Transformer family sweep done $(date) ===" >> $LOG
