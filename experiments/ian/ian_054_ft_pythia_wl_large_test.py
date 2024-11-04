import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/053_ft_pythia_wl"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-6.9b", "pythia-6.9b_clf_wl_s{}_v054"),
    ("EleutherAI/pythia-12b", "pythia-12b_clf_wl_s{}_v054"),
]

SEEDS = [0, 1, 2]

OVERRIDE_ARGS_LIST = [
    {
        "+model": base_model_name,
        "model.train_minibatch_size": 2,  # 2 replicates niki's 015 and 089a
        "training.force_name_to_save": target_model_name.format(seed),
        "training.seed": seed,
    }
    for seed in SEEDS  # This will run everything for a seed, then move to next seed
    for (
        base_model_name,
        target_model_name,
    ) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES
]
if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        gpu=3,
        override_args_list=OVERRIDE_ARGS_LIST,
        container_tag="fix-k8s-backoff-v3",
        memory="200G",
    )
