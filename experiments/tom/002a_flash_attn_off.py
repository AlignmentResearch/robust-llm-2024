# Test an eval with flash attention. Compare to 000d (launched by
# 000c_bfloat_off.py).
from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

FILE = Path(__file__)
EXPERIMENT_NAME = FILE.parent.name + "_" + FILE.stem
HYDRA_CONFIG = "tom/002_ft_pythia_imdb.yaml"

MODELS_AND_N_MAX_PARALLEL = [
    ("EleutherAI/pythia-160m", 1),
]


OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
            "model.dtype": "bfloat16",
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
        memory="30G",
        cpu=1,
        priority="normal-batch",
        container_tag="flash-attn",
    )
    EXPERIMENT_NAME = EXPERIMENT_NAME.replace("002a", "002b")
    EXPERIMENT_NAME = EXPERIMENT_NAME.replace("attn_off", "attn_on")
    for override in OVERRIDE_ARGS_LIST:
        override["model.attention_implementation"] = "flash_attention_2"
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="30G",
        cpu=1,
        priority="normal-batch",
        container_tag="flash-attn",
    )
