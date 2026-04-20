import matplotlib.pyplot as plt
import numpy as np


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


def plot_waterfall(dim, val, title="Waterfall Chart"):
    dim = list(dim)
    val = list(val)

    assert len(dim) == len(val), "dim and val must have the same length"
    assert len(dim) >= 1, "Need at least one bar"

    total = val[-1]
    step_dim = dim[:-1]
    step_val = val[:-1]

    cum = np.cumsum([0] + step_val[:-1]) if step_val else []

    BAR_COLOR = "#1434CB"
    POS_TOTAL = "#59A14F"
    NEG_TOTAL = "#E45756"
    TOTAL_COLOR = POS_TOTAL if total >= 0 else NEG_TOTAL
    CONNECTOR_COLOR = "#D3D3D3"
    BG_COLOR = "#F7F7F7"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # incremental bars
    for i, (d, v, start) in enumerate(zip(step_dim, step_val, cum)):
        ax.bar(i, v, bottom=start, color=BAR_COLOR, edgecolor="none")
        ax.text(
            i,
            start + v / 2,
            f"{v:+.2%}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
        )

    # total bar (last one)
    total_x = len(dim) - 1
    ax.bar(total_x, total, bottom=0, color=TOTAL_COLOR, edgecolor="none")
    ax.text(
        total_x,
        total / 2,
        f"{total:+.2%}",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )

    # connectors
    for i in range(len(step_val) - 1):
        y = cum[i] + step_val[i]
        ax.plot(
            [i + 0.4, i + 1 - 0.4],
            [y, y],
            linestyle="-",
            color=CONNECTOR_COLOR,
            linewidth=1,
        )

    if len(step_val) > 0:
        y = cum[-1] + step_val[-1]
        ax.plot(
            [len(step_val) - 1 + 0.4, total_x - 0.4],
            [y, y],
            linestyle="-",
            color=CONNECTOR_COLOR,
            linewidth=1,
        )

    # formatting
    labels = [d.replace("|", "\n") for d in dim]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, ha="center")

    ax.axhline(0, color="#AAAAAA", linewidth=1)
    ax.grid(axis="y", color="#EAEAEA", linewidth=1)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    ax.tick_params(axis="y", length=0, colors="#666666")
    ax.tick_params(axis="x", colors="#333333", labelsize=9)

    for label in ax.get_xticklabels():
        label.set_linespacing(1.2)

    ax.set_title(title, fontsize=13, weight="bold")

    plt.tight_layout()
    plt.show()
