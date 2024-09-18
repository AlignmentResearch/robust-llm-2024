import os

from tqdm import tqdm

from robust_llm.metrics.iterations_for_success import compute_all_ifs_metrics

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
for group in tqdm(GROUPS):
    asr_path = f"outputs/asr_{group}.csv"
    ifs_path = f"outputs/ifs_{group}.csv"
    if os.path.exists(ifs_path) and os.path.exists(asr_path):
        print(f"Skipping {ifs_path} & {asr_path}")
        continue
    asr_df, ifs_df = compute_all_ifs_metrics(
        group,
        n_models=MODELS,
        n_seeds=SEEDS,
        max_workers=MAX_WORKERS,
    )
    print(f"Saving to {asr_path}")
    asr_df.to_csv(asr_path, index=False)
    print(f"Saving to {ifs_path}")
    ifs_df.to_csv(ifs_path, index=False)
