import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_imdb"  # We replace the dataset in the overrides below.

DATASETS = [
    # Full dataset name, short name, whether to cut dataset size.
    # (if we cut, we use 50K for train and 5K for validation).
    # We cut only for yelp which has >500K training examples.
    # Other datasets are ~10K-100K.
    ("carblacac/twitter-sentiment-analysis", "twitter", False),
    ("climatebert/climate_detection", "climate", False),
    ("rotten_tomatoes", "rt", False),
    ("sst2", "sst2", False),
    ("yelp_polarity", "yelp", True),
]
PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-<DATASET_SHORT>-14m-mz-ada-v3"),
    ("EleutherAI/pythia-31m", "pythia-<DATASET_SHORT>-31m-mz-ada-v3"),
    ("EleutherAI/pythia-70m-deduped", "pythia-<DATASET_SHORT>-70m-mz-ada-v3"),
    ("EleutherAI/pythia-160m-deduped", "pythia-<DATASET_SHORT>-160m-mz-ada-v3"),
    ("EleutherAI/pythia-410m-deduped", "pythia-<DATASET_SHORT>-410m-mz-ada-v3"),
    ("EleutherAI/pythia-1b-deduped", "pythia-<DATASET_SHORT>-1b-mz-ada-v3"),
]

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.dataset_type": f"hf/{dataset_full}",
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.training.force_name_to_save": target_model_name.replace(
            "<DATASET_SHORT>", dataset_short
        ),
        "experiment.training.optimizer": "adafactor",
        "experiment.training.save_strategy": "no",
        "experiment.training.gradient_checkpointing": True,
        "experiment.environment.train_set_size": 50_000 if cut_dataset else None,
        "experiment.environment.validation_set_size": 5_000 if cut_dataset else None,
    }
    for (base_model_name, target_model_name) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES
    for (dataset_full, dataset_short, cut_dataset) in DATASETS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
    )
