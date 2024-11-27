METRICS = [
    "adversarial_eval/attack_success_rate",
]

SUMMARY_KEYS = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.training.seed",
    "experiment_yaml.training.adversarial.num_examples_to_generate_each_round",
    "experiment_yaml.dataset.n_train",
    "model_size",
]

PROJECT_NAME = "farai/robust-llm"

MODEL_NAMES = {
    "pythia": [
        "14m",
        "31m",
        "70m",
        "160m",
        "410m",
        "1b",
        "1.4b",
        "2.8b",
        "6.9b",
        "12b",
    ],
    "qwen": ["0.5B", "1.5B", "3B", "7B", "14B"],
}


MODEL_SIZES = {
    "clf": {
        "pythia": [
            7629056,
            17617408,
            44672000,
            123691008,
            353824768,
            908763136,  # 1b
            1311629312,  # 1.4b
            2646435840,  # 2.8b
            6650740736,  # 6.9b
            11586560000,  # 12b
        ],
        "qwen": [494034560, 1543717376, 3085942784, 7070626304, 13991476224],
    },
    "gen": {
        "qwen": [494032768, 1543714304, 7615616512, 3085938688, 14770033664],
    },
}


FINAL_PYTHIA_CHECKPOINT = 143_000
