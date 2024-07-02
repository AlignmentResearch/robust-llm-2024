#!/bin/bash

# Run the training script in the background
python robust_llm +experiment=Eval/pm_gcg.yaml dataset.n_val=1000 environment.logging_level=10 evaluation.evaluation_attack.n_its=10 evaluation.evaluation_attack.save_steps=1 run_name=oskar_001_interrupt_gcg_eval_diff_args &
# Sleep for a certain period to let the training start
sleep 60  # seconds

# Kill the training process to simulate interruption
echo "\nKilling process number $(pgrep -f 'python robust_llm')"
kill $(pgrep -f 'python robust_llm')

# # Restart the training process to verify resumption
python robust_llm +experiment=Eval/pm_gcg.yaml dataset.n_val=1000 environment.logging_level=10 evaluation.evaluation_attack.n_its=9 evaluation.evaluation_attack.save_steps=1 run_name=oskar_001_interrupt_gcg_eval_diff_args
