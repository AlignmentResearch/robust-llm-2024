# Rerun of 056 with ground_truth_label_fn working for pm.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/056_pm_gcg_robustness_metric"

BASE_MODEL_N_GPUS_MEMORY = [
    (
        "EleutherAI/pythia-14m",
        "AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3",
        1,
        "10G",
    ),
    (
        "EleutherAI/pythia-31m",
        "AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v3",
        1,
        "10G",
    ),
    (
        "EleutherAI/pythia-70m",
        "AlignmentResearch/robust_llm_pythia-tt-70m-mz-ada-v3",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-160m",
        "AlignmentResearch/robust_llm_pythia-tt-160m-mz-ada-v3",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-410m",
        "AlignmentResearch/robust_llm_pythia-tt-410m-mz-ada-v3",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-1b",
        "AlignmentResearch/robust_llm_pythia-tt-1b-mz-ada-v3",
        1,
        "30G",
    ),
    (
        "EleutherAI/pythia-1.4b",
        "AlignmentResearch/robust_llm_pythia-tt-1.4b-mz-ada-v3",
        1,
        "30G",
    ),
    (
        "EleutherAI/pythia-2.8b",
        "AlignmentResearch/robust_llm_pythia-tt-2.8b-mz-ada-v3",
        1,
        "40G",
    ),
]

OVERRIDE_TUPLES = [
    (
        {
            "+model": base_model,
            "model.name_or_path": model_name,
        },
        n_gpus,
        memory,
    )
    for (base_model, model_name, n_gpus, memory) in BASE_MODEL_N_GPUS_MEMORY
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cpu=8,
        priority="normal-batch",
    )
