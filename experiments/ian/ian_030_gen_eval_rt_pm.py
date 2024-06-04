# Compare to ian_017
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/030_gen_eval_rt_pm"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-14m-pm-gen-ian-nd", 2),
    ("AlignmentResearch/robust_llm_pythia-31m-pm-gen-ian-nd", 2),
    ("AlignmentResearch/robust_llm_pythia-70m-pm-gen-ian-nd", 1),
    ("AlignmentResearch/robust_llm_pythia-160m-pm-gen-ian-nd", 1),
    ("AlignmentResearch/robust_llm_pythia-410m-pm-gen-ian-nd", 1),
    ("AlignmentResearch/robust_llm_pythia-1b-pm-gen-ian-nd", 1),
    ("AlignmentResearch/robust_llm_pythia-2.8b-pm-gen-ian-nd", 1),
]


OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
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
        memory="50G",
        cpu=12,
        priority="high-batch",
    )
