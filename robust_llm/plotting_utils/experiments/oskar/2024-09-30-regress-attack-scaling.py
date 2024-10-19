# %%
import pandas as pd
import statsmodels.formula.api as smf

from robust_llm.wandb_utils.constants import MODEL_NAMES

pd.set_option("display.max_columns", 500)


# %%
def regress_attack_scaling(path: str):
    df = pd.read_csv(path)
    df = df.loc[df.asr.between(0.01, 0.99)]
    df["model"] = df.model_idx.apply(lambda i: MODEL_NAMES[i])
    reg = smf.ols(
        "logit_asr ~ np.log10(attack_flops_fraction_pretrain) + model", data=df
    ).fit()
    print(reg.summary())
    return reg.params["np.log10(attack_flops_fraction_pretrain)"]


# %%
print("IMDB")
grads = []
for round in (
    "round_0",
    "pretrain_fraction_1_bps",
    "pretrain_fraction_10_bps",
    "pretrain_fraction_50_bps",
    "final_round",
):
    print(f"Round: {round}")
    grad = regress_attack_scaling(
        f"/home/oskar/projects/robust-llm/plots/asr/gcg_gcg/imdb/{round}/attack_flops_fraction_pretrain/logit_asr/smoothing-1/data.csv"  # noqa
    )
    grads.append(grad)
# %%
print([grads[0] / grad for grad in grads])

# %%
