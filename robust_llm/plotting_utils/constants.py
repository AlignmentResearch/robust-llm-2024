from robust_llm.wandb_utils.constants import MODEL_SIZES

MODEL_PLOTTING_NAMES = [
    "7.6M",
    "17.6M",
    "44.7M",
    "123.7M",
    "353.8M",
    "908.8M",
    "1.3B",
    "2.6B",
    "6.7B",
    "11.6B",
]

assert len(MODEL_SIZES) == len(MODEL_PLOTTING_NAMES)
PLOTTING_NAME_DICT = dict(zip(MODEL_SIZES, MODEL_PLOTTING_NAMES))

# This is a multiplier we use to make up for the fact that we didn't
# log multi-GPU flops properly for the 12b model.
FUDGE_FOR_12B = 1.6

DEFAULT_SMOOTHING = 1
