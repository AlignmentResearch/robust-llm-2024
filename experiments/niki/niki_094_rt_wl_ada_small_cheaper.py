import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-29_niki_094"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-word-length-14m-niki-ada-v1", 4),
    ("AlignmentResearch/robust_llm_pythia-word-length-31m-niki-ada-v1", 2),
    ("AlignmentResearch/robust_llm_pythia-word-length-70m-niki-ada-v1", 1),
    ("AlignmentResearch/robust_llm_pythia-word-length-160m-niki-ada-v1", 1),
    ("AlignmentResearch/robust_llm_pythia-word-length-410m-niki-ada-v1", 1),
    ("AlignmentResearch/robust_llm_pythia-word-length-1b-niki-ada-v1", 1),
    ("AlignmentResearch/robust_llm_pythia-word-length-1.4b-niki-ada-v1", 1),
    ("AlignmentResearch/robust_llm_pythia-word-length-2.8b-niki-ada-v1", 1),
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
            "evaluation.batch_size": 32,
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="100G",
        cpu=12,
        priority="high-batch",
    )
