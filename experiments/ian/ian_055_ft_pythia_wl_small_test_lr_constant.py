import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/055_ft_pythia_wl_constant"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-14m_clf_wl_s{}_v055"),
    ("EleutherAI/pythia-31m", "pythia-31m_clf_wl_s{}_v055"),
    ("EleutherAI/pythia-70m", "pythia-70m_clf_wl_s{}_v055"),
]

SEEDS = [0, 1, 2, 3, 4]

OVERRIDE_ARGS_LIST = [
    {
        "+model": base_model_name,
        "model.train_minibatch_size": 64,
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
        gpu=1,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory="50G",
        container_tag="2024-08-07-backoff",
    )
