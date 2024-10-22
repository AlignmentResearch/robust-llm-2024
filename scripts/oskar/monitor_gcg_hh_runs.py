from datetime import datetime

import wandb

from robust_llm.wandb_utils.wandb_api_tools import get_tracking_data_for_runs


def search_wandb_runs():
    api = wandb.Api(timeout=60)

    # Format the date string in ISO 8601 format
    filter_date = datetime(2024, 10, 1).strftime("%Y-%m-%dT%H:%M:%SZ")

    runs = api.runs(
        "farai/robust-llm",
        filters={
            "$and": [
                {
                    "$or": [
                        {"group": "niki_173_adv_tr_gcg_helpful_small"},
                        {"group": "niki_174_adv_tr_gcg_harmless_small"},
                    ]
                },
                {"created_at": {"$gt": filter_date}},
            ]
        },
        order="+created_at",
    )
    return get_tracking_data_for_runs(runs)


if __name__ == "__main__":
    output_csv = "/home/oskar/projects/robust-llm/outputs/gcg_hh_runs.csv"

    df = search_wandb_runs()
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' has been created with {len(df)} entries.")
