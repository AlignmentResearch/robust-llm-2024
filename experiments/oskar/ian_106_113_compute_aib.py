import os

from tqdm import tqdm

from robust_llm.metrics.average_initial_breach import compute_all_aibs

GROUPS = [
    "ian_106_gcg_pythia_imdb",
    "ian_107_gcg_pythia_pm",
    "ian_108_gcg_pythia_wl",
    "ian_109_gcg_pythia_spam",
    "ian_110_rt_pythia_imdb",
    "ian_111_rt_pythia_pm",
    "ian_112_rt_pythia_wl",
    "ian_113_rt_pythia_spam",
]
SEEDS = 5
MODELS = 10
MAX_WORKERS = 4
SUB_WORKERS = 4
for group in tqdm(GROUPS):
    path = f"outputs/aibs_{group}.csv"
    if os.path.exists(path):
        print(f"Skipping {path}")
        continue
    metrics_df = compute_all_aibs(
        group,
        n_models=MODELS,
        n_seeds=SEEDS,
        max_workers=MAX_WORKERS,
        sub_workers=SUB_WORKERS,
    )
    print(f"Saving to {path}")
    metrics_df.to_csv(path, index=False)
