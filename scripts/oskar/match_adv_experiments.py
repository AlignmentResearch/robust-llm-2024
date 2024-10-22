import wandb

from robust_llm.wandb_utils.wandb_api_tools import get_tracking_data_for_runs


def search_wandb_runs():
    api = wandb.Api(timeout=60)
    runs = api.runs(
        "farai/robust-llm",
        filters={"config.hub_model_id": {"$regex": ".*_adv_tr.*"}, "state": "finished"},
    )
    return get_tracking_data_for_runs(runs)


if __name__ == "__main__":
    output_csv = (
        "/home/oskar/projects/robust-llm/outputs/adv_model_experiment_mapping.csv"
    )

    df = search_wandb_runs()
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' has been created with {len(df)} entries.")
