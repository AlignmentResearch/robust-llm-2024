import argparse

from robust_llm.wandb_utils.wandb_api_tools import zip_attack_data_tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip attack data tables.")
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the attack data tables to zip.",
        default=None,
    )
    args = parser.parse_args()
    zip_attack_data_tables(only_prefix=args.prefix)
