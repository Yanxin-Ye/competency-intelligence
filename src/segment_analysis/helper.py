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


def get_generation(year):
    if year >= 2013:
        return "Gen Alpha"
    elif year >= 1997:
        return "Gen Z"
    elif year >= 1981:
        return "Millennial"
    elif year >= 1965:
        return "Gen X"
    elif year >= 1946:
        return "Boomer"
    else:
        return "Unknown"
