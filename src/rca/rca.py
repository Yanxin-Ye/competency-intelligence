import pandas as pd
import etl
from pathlib import Path


def load_rca_data(target_merchants: str):
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "etl" / f"rca_{target_merchants}.csv"

    print(file_path)
    if file_path.exists():
        df_comparison = pd.read_csv(file_path)
    else:
        print("File does not exist, loading data from source...")
        df_comparison = etl.load_trans_data()
    df_combined = etl.preprocess(df_comparison, target_merchants)
    return df_combined


if __name__ == "__main__":
    TARGET_MERCHANTS = "fraud_Kilback LLC"

    df_combined = load_rca_data(TARGET_MERCHANTS)
    print("Data loaded successfully.")
    print(f"Data shape: {df_combined.shape}")
    print(f"Sample data:\n{df_combined.head()}")
