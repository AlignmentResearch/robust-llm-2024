#!/bin/bash

# Run the training script in the background
python robust_llm +experiment=Eval/imdb_red-team.yaml +model@evaluation.evaluation_attack.adversary=pythia-14m dataset.n_val=1000 environment.logging_level=10 evaluation.evaluation_attack.n_its=10 evaluation.evaluation_attack.save_steps=1 run_name=oskar_003_interrupt_lm_attack &
# Sleep for a certain period to let the training start
sleep 60  # seconds

# Kill the training process to simulate interruption
echo "\nKilling process number $(pgrep -f 'python robust_llm')"
kill $(pgrep -f 'python robust_llm')

# # Restart the training process to verify resumption
python robust_llm +experiment=Eval/imdb_red-team.yaml +model@evaluation.evaluation_attack.adversary=pythia-14m dataset.n_val=1000 environment.logging_level=10 evaluation.evaluation_attack.n_its=10 evaluation.evaluation_attack.save_steps=1 run_name=oskar_003_interrupt_lm_attack
