from pathlib import Path

SHARED_DATA_DIR = "/robust_llm_data"


MODEL_FAMILIES = [
    "gpt2",
    "llama2",
    "llama2-chat",
    "llama3",
    "llama3-chat",
    "vicuna",
    "pythia",
    "qwen1.5",
    "qwen2",
    "qwen1.5-chat",
    "qwen2-chat",
    "tinyllama",
    "mistralai",
    "gpt_neox",
    "pythia-chat",
    "gemma",
    "gemma-chat",
]


def get_save_root() -> str:
    if Path(SHARED_DATA_DIR).exists():
        return SHARED_DATA_DIR
    else:
        return "cache"
