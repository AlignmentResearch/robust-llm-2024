METRICS = [
    "adversarial_eval/pre_attack_accuracy",
    "adversarial_eval/post_attack_accuracy_including_original_mistakes",
    "adversarial_eval/attack_success_rate",
]

SUMMARY_KEYS = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.training.seed",
    "model_size",
]

PROJECT_NAME = "farai/robust-llm"

MODEL_SIZE_DICT = {
    "14m": 14_000_000,
    "31m": 31_000_000,
    "70m": 70_000_000,
    "160m": 160_000_000,
    "410m": 410_000_000,
    "1b": 1_000_000_000,
    "1.4b": 1_400_000_000,
    "2.8b": 2_800_000_000,
    "6.9b": 6_900_000_000,
    "12b": 12_000_000_000,
}

FINAL_PYTHIA_CHECKPOINT = 143_000
