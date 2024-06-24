#!/bin/bash

# Run the training script in the background
python robust_llm +experiment=AdvTraining/imdb_random-token ++training.save_steps=1 ++training.logging_steps=10 ++training.hardcoded_output_dir=outputs/signal_interrupt_adv ++training.model_save_path_prefix_or_hf=outputs ++training.adversarial.num_adversarial_training_rounds=1000 ++environment.logging_level=10  ++training.force_name_to_save=dummy_hub_model_id &

# Sleep for a certain period to let the training start
sleep 60  # seconds

# Kill the training process to simulate interruption
echo "\nKilling process number $(pgrep -f 'python robust_llm')"
kill $(pgrep -f 'python robust_llm')

# Restart the training process to verify resumption
python robust_llm +experiment=AdvTraining/imdb_random-token ++training.save_steps=1 ++training.logging_steps=10 ++training.hardcoded_output_dir=outputs/signal_interrupt_adv ++training.model_save_path_prefix_or_hf=outputs ++training.adversarial.num_adversarial_training_rounds=1000 ++environment.logging_level=10  ++training.force_name_to_save=dummy_hub_model_id
