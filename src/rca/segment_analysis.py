import etl
from itertools import combinations
from config import DIMENSION_COLS


def grpby_dim_val(df, dims, target_col):
    kv = {}

    # make dims always a list
    if isinstance(dims, str):
        dims = [dims]

    grouped = (
        df[dims + [target_col]]
        .groupby(dims, dropna=False)[target_col]
        .sum()
        .reset_index()
    )

    for row in grouped.to_dict("records"):
        k_parts = [f"{dim} = {row[dim]}" for dim in dims]
        k = ", ".join(k_parts)
        v = row[target_col]
        kv[k] = v

    return kv


def exhaustive_eval(df_combined, target_col):
    res = {}
    n = len(DIMENSION_COLS)
    for k in range(1, n + 1):
        for comb in combinations(DIMENSION_COLS, k):
            res.update(grpby_dim_val(df_combined, list(comb), target_col))
    sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    return sorted_res


if __name__ == "__main__":
    TARGET_MERCHANTS = "fraud_Kilback LLC"
    target_col = "amt_growth_ctc_diff"

    df_combined = etl.load_rca_data(TARGET_MERCHANTS)
    print("Data loaded successfully.")
    print(f"Data shape: {df_combined.shape}")
    print(f"Sample data:\n{df_combined.head()}")
    print(f"Total growth difference: {df_combined[target_col].sum():.2%}")

    # Option1: Perform exhaustive evaluation to find top dimension values contributing to growth difference
    sorted_res = exhaustive_eval(df_combined, target_col)
    total = df_combined[target_col].sum()
    for i, (k, v) in enumerate(sorted_res[:20], 1):
        print(f"{i:>2}. {k:<50} | {v:>8.2%} | contribution: {v/total:>6.2%}")
