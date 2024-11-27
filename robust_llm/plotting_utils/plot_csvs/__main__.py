import argparse

from robust_llm.plotting_utils.plot_csvs.adv_training import main as adv_training_main
from robust_llm.plotting_utils.plot_csvs.adv_training_transfer import (
    main as adv_training_transfer_main,
)
from robust_llm.plotting_utils.plot_csvs.asr_adv_training import (
    main as asr_adv_training_main,
)
from robust_llm.plotting_utils.plot_csvs.asr_finetuned import main as asr_finetuned_main
from robust_llm.plotting_utils.plot_csvs.asr_slopes import main as asr_slopes_main
from robust_llm.plotting_utils.plot_csvs.finetuned import main as finetuned_main
from robust_llm.plotting_utils.plot_csvs.offense_defense import (
    main as offense_defense_main,
)
from robust_llm.plotting_utils.plot_csvs.post_adv_training import (
    main as post_adv_training_main,
)

PLOTS = [
    "adv_training_transfer",
    "adv_training",
    "asr_adv_training",
    "asr_finetuned",
    "asr_slopes",
    "finetuned",
    "finetuned_qwen",
    "offense_defense",
    "post_adv_training",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run plotting utilities with a specified style."
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Style to be passed to each plotting function",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        choices=PLOTS,
        help="List of plot types to generate",
        default=PLOTS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if "adv_training_transfer" in args.plots:
        print("Transfer plots...")
        adv_training_transfer_main(args.style)
    if "adv_training" in args.plots:
        print("Regular adversarial training plots...")
        adv_training_main(args.style)
    if "asr_adv_training" in args.plots:
        print("Attack scaling for adversarially trained models...")
        asr_adv_training_main(args.style)
    if "asr_finetuned" in args.plots:
        print("Attack scaling for finetuned models...")
        asr_finetuned_main(args.style)
    if "asr_slopes" in args.plots:
        print("Attack scaling slopes")
        asr_slopes_main(args.style)
    if "finetuned" in args.plots:
        print("Finetuning plots...")
        finetuned_main(args.style)
    if "offense_defense" in args.plots:
        print("Offense-defense plots...")
        offense_defense_main(args.style)
    if "post_adv_training" in args.plots:
        print("Finetuning-like plots")
        post_adv_training_main(args.style)
