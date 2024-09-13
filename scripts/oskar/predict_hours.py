# %%
import itertools

import pandas as pd
import statsmodels.formula.api as smf

pd.set_option("display.max_columns", 50)
# %%
output_csv = "/home/oskar/projects/robust-llm/outputs/adv_model_experiment_mapping.csv"
df = pd.read_csv(output_csv)
print(df.head())
# %%
model = smf.ols(
    "h100_hours ~ num_parameters + attack + dataset + num_adversarial_training_rounds",
    data=df,
)
results = model.fit()
print(results.summary())
# %%
model_sizes = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
datasets = ["imdb", "pm", "wl", "spam", "harmless", "helpful"]
seeds = range(5)  # 0 to 4
attacks = ["gcg", "rt"]
# %%
all_tuples = itertools.product(model_sizes, datasets, seeds, attacks)
remaining_tuples = set(all_tuples) - set(
    zip(df.base_model, df.dataset, df.adv_seed, df.attack)
)
print(len(remaining_tuples))
# %%
rem_df = pd.DataFrame(
    remaining_tuples,
    columns=["base_model", "dataset", "adv_seed", "attack"],  # type: ignore
)
rem_df["num_parameters"] = (
    rem_df.base_model.str.split("-")
    .str[-1]
    .str.replace("m", "e6")
    .str.replace("b", "e9")
    .astype(float)
)
# N_ADV_TR_ROUNDS = [953, 413, 163, 59, 21, 8, 6, 3, 1, 1]
rem_df["num_adversarial_training_rounds"] = (
    rem_df.base_model.map(
        {
            "14m": 953,
            "31m": 413,
            "70m": 163,
            "160m": 59,
            "410m": 21,
            "1b": 8,
            "1.4b": 6,
            "2.8b": 3,
            "6.9b": 1,
            "12b": 1,
        }
    ).clip(5, 60)
    + 1
)
rem_df["pred_hours"] = results.predict(rem_df)
print(rem_df.pred_hours.sum())
# %%
