# Test an eval with flash attention on Qwen. Compare to 000f (launched by
# 000e_bfloat_off.py).
from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

FILE = Path(__file__)
EXPERIMENT_NAME = FILE.parent.name + "_" + FILE.stem
HYDRA_CONFIG = "tom/000_gen_pm_ihateyou_gcg"

MODELS_AND_N_MAX_PARALLEL = [
    ("Qwen/Qwen1.5-0.5B-Chat", 2),
    ("Qwen/Qwen1.5-1.8B-Chat", 2),
]


OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model": model,
            "model.dtype": "bfloat16",
            "model.attention_implementation": "flash_attention_2",
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
        cpu=1,
        priority="normal-batch",
        container_tag="flash-attn",
    )
