#!/bin/bash

# Set the working directory to the directory of this script
cd "$(dirname "$0")"

if [ "$1" == "clear-cache" ]; then
    rm -rf cache/get-metrics-adv-training
fi
python finetuning_vs_adv_training_new_data.py # Figure 1
python ian_106_asr.py # Figure 2 top row. Dependency: experiments/oskar/ian_106_113_compute_ifs.py
python adv_training_tom_005a_metrics.py # Figure 2 bottom row
python adv_training_tom_005a_non_transfer.py # Figure 3
python adv_training_tom_005a.py # Figures 4 and 5
