import wandb

from robust_llm.file_utils import compute_repo_path
from robust_llm.wandb_utils.wandb_api_tools import parse_runs_to_dataframe


def search_wandb_runs():
    api = wandb.Api(timeout=60)
    runs = api.runs(
        "farai/robust-llm",
        filters={
            "display_name": {"$regex": "tom-.*-eval-niki-.*-gcg-.*"},
            "state": "finished",
        },
    )
    return parse_runs_to_dataframe(runs)


if __name__ == "__main__":
    output_csv = f"{compute_repo_path()}/outputs/tom_adv_eval_experiments.csv"
    df = search_wandb_runs()
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' has been created with {len(df)} entries.")
