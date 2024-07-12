# Reduced batch size to 16 to improve training.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/029a_gen_ft_pythia_pm"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES_AND_PARALLEL = [
    ("EleutherAI/pythia-14m", "pythia-14m-pm_1.2.0-gen-ian-nd-v2", 1),
    ("EleutherAI/pythia-31m", "pythia-31m-pm_1.2.0-gen-ian-nd-v2", 1),
    ("EleutherAI/pythia-70m", "pythia-70m-pm_1.2.0-gen-ian-nd-v2", 1),
    ("EleutherAI/pythia-160m", "pythia-160m-pm_1.2.0-gen-ian-nd-v2", 1),
    ("EleutherAI/pythia-410m", "pythia-410m-pm_1.2.0-gen-ian-nd-v2", 1),
    ("EleutherAI/pythia-1b", "pythia-1b-pm_1.2.0-gen-ian-nd-v2", 1),
    ("EleutherAI/pythia-2.8b", "pythia-2.8b-pm_1.2.0-gen-ian-nd-v2", 1),
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "+model": base_model_name,
            # We have to set a fixed minibatch in Python because otherwise it's
            # overridden by +model.
            "model.train_minibatch_size": 16,
            "training.force_name_to_save": target_model_name,
        },
        n_max_parallel,
    )
    for (
        base_model_name,
        target_model_name,
        n_max_parallel,
    ) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES_AND_PARALLEL  # noqa: E501
]
OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        n_max_parallel=N_MAX_PARALLEL,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory="50G",
    )
