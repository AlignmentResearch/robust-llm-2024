# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def calculate_logit_asr(log_iteration_flops, log_params, params):
    return (
        params["Intercept"]
        + params["log_iteration_flops"] * log_iteration_flops
        + params["log_params"] * log_params
        + params["log_iteration_flops:log_params"] * log_iteration_flops * log_params
    )


# %%
def regress_attack_scaling(path: str):
    df = pd.read_csv(path)
    df = df.loc[np.isfinite(df.logit_asr)]
    df["log_params"] = np.log10(df.num_params)
    df["log_iteration_flops"] = np.log10(df.iteration_x_flops)
    df["pretrain_compute_percent"] = df.apply(
        lambda x: x.iteration_x_flops / float(x.pretrain_compute), axis=1
    )
    gradients = dict()
    reg = None
    for num_params, params_df in df.groupby("num_params"):
        reg = smf.ols("logit_asr ~ pretrain_compute_percent", data=params_df).fit()
        gradients[num_params] = reg.params["pretrain_compute_percent"]
    grad_df = pd.DataFrame(
        gradients.items(), columns=["num_params", "gradient"]  # type: ignore
    )
    grad_df.plot(
        x="num_params",
        y="gradient",
        logx=True,
        title="Gradient of attacking scaling plot (figure 2)",
    )
    if reg is not None:
        print(reg.summary())


# %%
print("Spam")
regress_attack_scaling(
    "/home/oskar/projects/robust-llm/plots/asr/gcg/spam/iteration_x_flops/logit_asr/data.csv"  # noqa
)
# %%
print("IMDB")
regress_attack_scaling(
    "/home/oskar/projects/robust-llm/plots/asr/gcg/imdb/iteration_x_flops/logit_asr/data.csv"  # noqa
)

# %%
