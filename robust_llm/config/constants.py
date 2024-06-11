import typing
from typing import Literal

SHARED_DATA_DIR = "/robust_llm_data"
# mistralai is for the paraphrase defense, but is not actually
# supported yet.
ModelFamily = Literal[
    "gpt2",
    "llama2",
    "pythia",
    "qwen",
    "qwen-chat",
    "tinyllama",
    "mistralai",
]
MODEL_FAMILIES = typing.get_args(ModelFamily)
