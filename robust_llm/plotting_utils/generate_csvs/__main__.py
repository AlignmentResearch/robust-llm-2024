import argparse

from robust_llm.plotting_utils.generate_csvs.adv_training import (
    main as adv_training_main,
)
from robust_llm.plotting_utils.generate_csvs.adv_training_transfer import (
    main as adv_training_transfer_main,
)
from robust_llm.plotting_utils.generate_csvs.asr_adv_training import (
    main as asr_adv_training_main,
)
from robust_llm.plotting_utils.generate_csvs.asr_finetuned import (
    main as asr_finetuned_main,
)
from robust_llm.plotting_utils.generate_csvs.finetuned import main as finetuned_main
from robust_llm.plotting_utils.generate_csvs.offense_defense import (
    main as offense_defense_main,
)
from robust_llm.plotting_utils.generate_csvs.post_adv_training import (
    main as post_adv_training_main,
)

EXPERIMENTS = [
    "adv_training_transfer",
    "adv_training",
    "asr_adv_training",
    "asr_finetuned",
    "finetuned",
    "offense_defense",
    "post_adv_training",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSVs for various experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        help="List of experiments to run (default: all)",
        default=EXPERIMENTS,
    )
    args = parser.parse_args()

    if "adv_training_transfer" in args.experiments:
        print("Transfer experiments...")
        adv_training_transfer_main()
    if "adv_training" in args.experiments:
        print("Regular adversarial training experiments...")
        adv_training_main()
    if "asr_adv_training" in args.experiments:
        print("Attack scaling for adversarially trained models...")
        asr_adv_training_main()
    if "asr_finetuned" in args.experiments:
        print("Attack scaling for finetuned models...")
        asr_finetuned_main()
    if "finetuned" in args.experiments:
        print("Finetuning experiments...")
        finetuned_main()
    if "offense_defense" in args.experiments:
        print("Offense-defense experiments...")
        offense_defense_main()
    if "post_adv_training" in args.experiments:
        print("Finetuning-like experiments for adversarially trained models...")
        post_adv_training_main()


if __name__ == "__main__":
    main()
