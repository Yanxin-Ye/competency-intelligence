from pathlib import Path

import numpy as np
import pandas as pd
from config import (
    DATA_API,
    DIMENSION_COLS,
    END_DATE_LY,
    END_DATE_TY,
    RANDOM_SEED,
    START_DATE_LY,
    START_DATE_TY,
    STATE_TO_MARKET,
)
from helper import get_generation


def load_trans_data():
    # Load dataset from huggingface, do some basic ETL to create a comparison dataset with TY and LY amounts by merchant and category
    splits = {
        "train": "credit_card_transaction_train.csv",
        "test": "credit_card_transaction_test.csv",
    }
    df_all = pd.read_csv(DATA_API + splits["train"], index_col=0)
    # ETL
    df_all = df_all.assign(
        trans_date_trans_time=lambda x: pd.to_datetime(x["trans_date_trans_time"]),
        trans_date=lambda x: x["trans_date_trans_time"].dt.strftime("%Y-%m-%d"),
    )
    df = (
        df_all[
            (
                (df_all["trans_date"] >= START_DATE_TY)
                & (df_all["trans_date"] <= END_DATE_TY)
            )
            | (
                (df_all["trans_date"] >= START_DATE_LY)
                & (df_all["trans_date"] <= END_DATE_LY)
            )
        ]
        .assign(
            dob=lambda x: pd.to_datetime(x["dob"]),
            generation=lambda x: x["dob"].dt.year.map(get_generation),
            market=lambda x: x["state"].map(STATE_TO_MARKET),
        )
        .copy()
    )

    df_comparison = (
        df.assign(
            yr_flag=lambda x: x["trans_date"].apply(
                lambda d: (
                    "TY"
                    if START_DATE_TY <= d <= END_DATE_TY
                    else ("LY" if START_DATE_LY <= d <= END_DATE_LY else None)
                )
            )
        )
        .groupby(DIMENSION_COLS + ["yr_flag", "merchant"], as_index=False)["amt"]
        .sum()
        .pivot_table(
            index=DIMENSION_COLS + ["merchant"],
            columns="yr_flag",
            values="amt",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"TY": "amt_ty", "LY": "amt_ly"})
        .rename_axis(None, axis=1)
    )

    # This data does not have any growth. Add some random variation to make it more interesting.
    # Add more variation to amt_ty and amt_ly for bigger growth

    np.random.seed(RANDOM_SEED)
    random_factors_ty = np.clip(np.random.normal(0.3, 25, len(df_comparison)), -1, 3)
    random_factors_ly = np.clip(np.random.normal(0, 2, len(df_comparison)), -0.5, 0.5)

    df_comparison["amt_ty"] *= 1 + random_factors_ty

    df_comparison["amt_ly"] *= 1 + random_factors_ly

    return df_comparison


def preprocess(df_comparison, target_merchants, save_data=True):
    assert (
        target_merchants in df_comparison["merchant"].values
    ), f"Target merchant {target_merchants} not found in data"
    assert (
        "category" in df_comparison.columns
    ), "DataFrame must contain 'category' column"
    df_target = df_comparison[df_comparison["merchant"] == target_merchants].copy()
    df_target = calculate_growth(df_target)

    target_industry = get_target_industry(df_comparison, target_merchants)
    df_peer = df_comparison[
        (df_comparison["merchant"] != target_merchants)
        & (df_comparison["category"].isin(target_industry))
    ].copy()

    peer_agg_metrics = {
        "amt_ty": ["mean"],
        "amt_ly": ["mean"],
    }

    peer_agg = (
        df_peer.groupby(DIMENSION_COLS)[["amt_ty", "amt_ly"]]
        .agg(peer_agg_metrics)
        .reset_index()
    )

    peer_agg.columns = [
        (
            f"{col[0]}_{col[1].lower()}".replace("_mean", "_peer")
            if col[1]
            else col[0].lower()
        )
        for col in peer_agg.columns
    ]

    total_amt_ly_peer = peer_agg["amt_ly_peer"].sum()
    peer_agg = peer_agg.assign(
        amt_diff_peer=lambda x: x["amt_ty_peer"] - x["amt_ly_peer"],
        amt_growth_ctc_peer=lambda x: x["amt_diff_peer"] / total_amt_ly_peer,
        merchant=f"ex-{target_merchants}",
    )

    peer_agg.head()
    df_combined = pd.merge(
        df_target,
        peer_agg,
        on=DIMENSION_COLS,
        how="outer",
        suffixes=("_target", "_peer"),
    )
    df_combined["amt_growth_ctc_diff"] = df_combined["amt_growth_ctc"].sub(
        df_combined["amt_growth_ctc_peer"], fill_value=0
    )

    print(f"Average growth of target: {df_combined['amt_growth_ctc'].sum():.2%}")
    print(f"Average growth of peers: {df_combined['amt_growth_ctc_peer'].sum():.2%}")
    print(
        f"Average growth difference between target and peers: {df_combined['amt_growth_ctc_diff'].sum():.2%}"
    )
    if save_data:
        # Save the combined data to a CSV file
        save_file(df_combined, f"etl/rca_{target_merchants}.csv")

    return df_combined


def get_target_industry(df_comparison, target_merchants):
    target_industry = (
        df_comparison[df_comparison["merchant"] == target_merchants]["category"]
        .drop_duplicates()
        .to_list()
    )
    print(f"Target industry of {target_merchants}: {target_industry}")
    return target_industry


def calculate_growth(df):
    assert (
        "amt_ty" in df.columns and "amt_ly" in df.columns
    ), "DataFrame must contain 'amt_ty' and 'amt_ly' columns"
    df["amt_diff"] = df["amt_ty"] - df["amt_ly"]
    total_amt_diff = df["amt_diff"].sum()

    total_amt_ly = df["amt_ly"].sum()
    print(f"Total amount in LY: {total_amt_ly}")
    print(f"Total amount difference: {total_amt_diff}")
    print(f"Total growth: {total_amt_diff / total_amt_ly:.2%}")
    df["amt_growth_ctc"] = 1 / total_amt_ly * df["amt_diff"]

    return df


def save_file(df, filename):

    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_rca_data(target_merchants: str):
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "etl" / f"rca_{target_merchants}.csv"
    if file_path.exists():
        print(f"File exists, loading data from {file_path}...")
        df_combined = pd.read_csv(file_path)
    else:
        print("File does not exist, loading data from source...")
        df_comparison = load_trans_data()
        df_combined = preprocess(df_comparison, target_merchants)
    return df_combined


if __name__ == "__main__":

    TARGET_MERCHANTS = (
        "fraud_Kilback LLC"  # A merchant with good growth that beat peers.
    )
    TARGET_MERCHANTS = "fraud_Champlin, Rolfson and Connelly"  # A merchant with negative growth that underperforms peers.

    df_comparison = load_trans_data()
    print(f"Data loaded with shape: {df_comparison.shape}")
    print(f"Sample data:\n{df_comparison.head()}")

    # print(f"Sort merchants by total amount in TY:")
    # df_comparison["amt_diff"] = df_comparison["amt_ty"] - df_comparison["amt_ly"]
    # merchant_ty_totals = (
    #     df_comparison.groupby("merchant")["amt_diff"].sum().sort_values(ascending=True)
    # )
    # print(merchant_ty_totals.head(10))

    df_combined = preprocess(df_comparison, TARGET_MERCHANTS)
    print(f"Cleaned data saved with shape: {df_combined.shape}")
    print(f"Sample cleaned data:\n{df_combined.head()}")
