# %%
import matplotlib.pyplot as plt
import pandas as pd

GROUP_NAME = "ian_103a_gcg_pythia_helpful"
OUTPUTS_DIR = "../../../../outputs"
# %%
MODEL_NAMES = [
    "pythia-14m",
    "pythia-31m",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]
# %%
df = pd.read_csv(f"{OUTPUTS_DIR}/aibs_{GROUP_NAME}.csv", names=["model", "seed", "aib"])
print(df)
# %%

fig, ax = plt.subplots()
fig.suptitle(f"Average Initial Breach metric for {GROUP_NAME}")
ys = df.groupby("model")["aib"].mean()
stds = df.groupby("model")["aib"].std() / df.groupby("model")["aib"].count() ** 0.5
xs = list(range(df.model.max() + 1))
ax.plot(xs, ys)
ax.errorbar(xs, ys, yerr=stds, fmt="o")
ax.set_xlabel("Model")
ax.set_ylabel("Average Initial Breach")
ax.set_xticks(xs)
ax.set_xticklabels(MODEL_NAMES, rotation=45)

# %%
