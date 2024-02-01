import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_adv_eval_imdb"

MODELS = [
    "AlignmentResearch/robust_llm_pythia-imdb-14m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-31m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-70m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-160m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-410m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-1b-mz-v1",
]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
    }
    for model in MODELS
]

if __name__ == "__main__":
    run_multiple(EXPERIMENT_NAME, HYDRA_CONFIG, OVERRIDE_ARGS_LIST, memory="60G")
