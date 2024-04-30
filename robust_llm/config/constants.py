import typing
from typing import Literal

SHARED_DATA_DIR = "/robust_llm_data"
ModelFamily = Literal["gpt2", "pythia", "bert"]
MODEL_FAMILIES = typing.get_args(ModelFamily)
