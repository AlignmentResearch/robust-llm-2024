import argparse
import os

from datasets.utils.logging import disable_progress_bar
from tqdm import tqdm

from robust_llm.metrics.iterations_for_success import compute_all_ifs_metrics


def main(reverse: bool = False, max_workers: int = 12):
    disable_progress_bar()
    GROUPS = [
        "oskar_019a_infix_eval_adv_tr_gcg_imdb_small",
        "oskar_019b_suffix_eval_adv_tr_gcg_imdb_small",
        "ian_102a_gcg_pythia_harmless",
        "ian_103a_gcg_pythia_helpful",
        "ian_104a_rt_pythia_harmless",
        "ian_105a_rt_pythia_helpful",
        "ian_106_gcg_pythia_imdb",
        "ian_107_gcg_pythia_pm",
        "ian_108_gcg_pythia_wl",
        "ian_109_gcg_pythia_spam",
        "ian_110_rt_pythia_imdb",
        "ian_111_rt_pythia_pm",
        "ian_112_rt_pythia_wl",
        "ian_113_rt_pythia_spam",
        "tom_005a_eval_niki_149_gcg",
        "tom_006_eval_niki_150_gcg",
        "tom_007_eval_niki_152_gcg",
        "tom_007_eval_niki_152a_gcg",
        "tom_008_eval_niki_152_gcg_infix90",
        "tom_008_eval_niki_152a_gcg_infix90",
        "tom_009_eval_niki_170_gcg",
        "tom_010_eval_niki_170_gcg_infix90",
        "niki_iki_eval_niki_182_gcg",
    ]
    if reverse:
        GROUPS = list(reversed(GROUPS))
    os.makedirs("outputs", exist_ok=True)
    for group in tqdm(GROUPS):
        asr_path = f"cache_csvs/asr_{group}.csv"
        ifs_path = f"cache_csvs/ifs_{group}.csv"
        if os.path.exists(ifs_path) and os.path.exists(asr_path):
            print(f"Skipping {ifs_path} & {asr_path}")
            continue
        asr_df, ifs_df = compute_all_ifs_metrics(
            group,
            max_workers=max_workers,
        )
        print(f"Saving to {asr_path}")
        asr_df.to_csv(asr_path, index=False)
        print(f"Saving to {ifs_path}")
        ifs_df.to_csv(ifs_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reverse", action="store_true", help="Reverse the order of the groups"
    )
    parser.add_argument(
        "--max_workers", type=int, default=12, help="Number of workers to use"
    )
    args = parser.parse_args()
    main(reverse=args.reverse, max_workers=args.max_workers)
