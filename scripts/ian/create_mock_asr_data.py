"""Script from Claude."""

import re

import numpy as np
import pandas as pd

# Read the original CSV file
original_df = pd.read_csv("outputs/mock_adv_data.csv")


# Function to extract seed_idx from model_name_or_path
def extract_seed_idx(model_name):
    match = re.search(r"t-(\d+)", model_name)
    return int(match.group(1)) if match else 0


# Create a list to store the new data
new_data = []

# Get unique combinations of model_size, model_idx, seed_idx, and adv_training_round
original_df["seed_idx"] = original_df["model_name_or_path"].apply(extract_seed_idx)
unique_combinations = original_df[
    ["model_size", "model_idx", "seed_idx", "adv_training_round"]
].drop_duplicates()


# Function to generate mock ASR values
def generate_asr_values(num_iterations):
    x = np.linspace(0, 1, num_iterations)
    base_asr = 1 / (1 + np.exp(-10 * (x - 0.5)))  # Sigmoid function
    noise = np.random.normal(0, 0.05, num_iterations)
    asr = np.clip(base_asr + noise, 0, 1)
    return asr


# Generate new data
for _, row in unique_combinations.iterrows():
    asr_values = generate_asr_values(129)
    for iteration in range(129):
        new_data.append(
            {
                "model_size": row["model_size"],
                "model_idx": row["model_idx"],
                "seed_idx": row["seed_idx"],
                "adv_training_round": row["adv_training_round"],
                "iteration": iteration,
                "asr": asr_values[iteration],
            }
        )

# Create a new DataFrame with the generated data
new_df = pd.DataFrame(new_data)

# Write the new DataFrame to a CSV file
new_df.to_csv("outputs/asr_mock_attack_data.csv", index=False)

print("Mock dataset has been generated and saved as 'asr_mock_attack_data.csv'")
