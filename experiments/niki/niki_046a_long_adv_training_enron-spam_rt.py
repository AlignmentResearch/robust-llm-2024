import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "niki/2024-07-11_niki_046a"

PYTHIA_NAMES_AND_MAX_N_PARALLEL = [
    ("EleutherAI/pythia-1b", 1),
]
SEEDS = [2]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": base_model_name,
            # Save to HF since we want to use these random token hardened models for
            # the beam search attack as well
            "training.force_name_to_save": f"{base_model_name[11:]}_{EXP_PREFIX}_enronspam_random-token-1280_seed-{seed}",  # noqa: E501
            # Defense uses random token 1280, attack uses gcg 10 iterations
            "training.seed": seed,
        },
        max_n_parallel,
    )
    for base_model_name, max_n_parallel in PYTHIA_NAMES_AND_MAX_N_PARALLEL
    for seed in SEEDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="60G",
        priority="normal-batch",
    )
