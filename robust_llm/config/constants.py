from enum import Enum

SHARED_DATA_DIR = "/robust_llm_data"


# mistralai is for the paraphrase defense, but is not actually
# supported yet.
class ModelFamily(Enum):
    GPT2 = 1
    LLAMA2 = 2
    PYTHIA = 3
    QWEN1_5 = 4
    QWEN2 = 5
    QWEN1_5_CHAT = 6
    QWEN2_CHAT = 7
    TINYLLAMA = 8
    MISTRALAI = 9
    GPT_NEOX = 10

    @staticmethod
    def from_string(string: str) -> "ModelFamily":
        return MODEL_FAMILY_STRINGS[string]


MODEL_FAMILY_STRINGS = {
    "gpt2": ModelFamily.GPT2,
    "llama2": ModelFamily.LLAMA2,
    "pythia": ModelFamily.PYTHIA,
    "qwen1.5": ModelFamily.QWEN1_5,
    "qwen2": ModelFamily.QWEN2,
    "qwen1.5-chat": ModelFamily.QWEN1_5_CHAT,
    "qwen2-chat": ModelFamily.QWEN2_CHAT,
    "tinyllama": ModelFamily.TINYLLAMA,
    "mistralai": ModelFamily.MISTRALAI,
    "gpt_neox": ModelFamily.GPT_NEOX,
}
MODEL_FAMILIES = list(MODEL_FAMILY_STRINGS.keys())
