import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from segment_analysis import etl
from segment_analysis.tree_search import TreeForest

load_dotenv()
print(os.getenv("GROQ_API_KEY"))


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


@tool
def run_segment_analysis_tree_search(
    entity: str, target_col: str, dim_cols: list
) -> str:
    """Run ETL process."""
    # your ETL logic
    df_combined = etl.load_rca_data(entity)
    forest = TreeForest(
        df_combined,
        target_col,
        dim_cols,
        n_trees=3,
        max_coverage=0.2,
        max_depth=4,
    )
    forest.construct_forest()

    print("\nLearned key drivers from the forest:")
    dim, val = forest.collect_key_drivers()
    peer_score = df_combined["amt_growth_ctc_peer"].sum()
    target_score = df_combined["amt_growth_ctc"].sum()
    print(f"Peer growth contribution: {peer_score:.2%}")
    print(f"Target growth contribution: {target_score:.2%}")
    dim = ["Peer Benchmark"] + dim + [f"Target ({TARGET_MERCHANTS})"]
    val = [peer_score] + val + [target_score]
    print("\nKey drivers and their contributions:")

    prompt = f"""\nLearned key drivers from the forest:\n
    Peer growth contribution: {peer_score:.2%}
    Target growth contribution: {target_score:.2%}
    """

    for d, v in zip(dim, val):
        print(f"{d:<30} | {v:>8.2%} | contribution: {v/target_score:>6.2%}")
        prompt += f"{d:<30} | {v:>8.2%} | contribution: {v/target_score:>6.2%}\n"

    # plot_waterfall(dim, val, title=f"Segment Analysis for {TARGET_MERCHANTS}")

    return prompt


if __name__ == "__main__":
    TARGET_MERCHANTS = "fraud_Kilback LLC"
    target_col = "amt_growth_ctc_diff"
    dim_cols = [
        "generation",
        "gender",
        "category",
        "state",
        "market",
    ]
    p = run_segment_analysis_tree_search(TARGET_MERCHANTS, target_col, dim_cols)
    print(p)
    # response = llm.invoke("Do you work?")
    # print(response.content)
