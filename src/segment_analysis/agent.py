import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
import etl
from tree_search import TreeForest

load_dotenv()
print(os.getenv("GROQ_API_KEY"))


def _run_segment_analysis_tree_search(
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
    dim = ["Peer Benchmark"] + dim + [f"Target ({TARGET_MERCHANTS})"]
    val = [peer_score] + val + [target_score]

    prompt = f"""\nLearned key drivers from the forest:\n
    Peer growth contribution: {peer_score:.2%}
    Target growth contribution: {target_score:.2%}
    The target merchant, {TARGET_MERCHANTS}, is {"" if target_score == 0 else f"{'outperforming' if target_score > peer_score else 'underperforming'} their peers"} by {target_score-peer_score:.2%} in terms of growth contribution. The key drivers and their contributions are as follows:\n
    """

    for d, v in zip(dim, val):
        print(f"{d:<30} | {v:>8.2%} | contribution: {v/target_score:>6.2%}")
        prompt += f"{d:<30} | {v:>8.2%} | contribution: {v/target_score:>6.2%}\n"

    # plot_waterfall(dim, val, title=f"Segment Analysis for {TARGET_MERCHANTS}")
    print(prompt)

    return prompt


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


if __name__ == "__main__":
    # TARGET_MERCHANTS = "fraud_Kilback LLC" # A randomly selected merchant
    # TARGET_MERCHANTS = "fraud_Wolf Inc"  # A merchant with good growth that beat peers.
    TARGET_MERCHANTS = "fraud_Champlin, Rolfson and Connelly"  # A merchant with negative growth that underperforms peers.
    target_col = "amt_growth_ctc_diff"
    dim_cols = [
        "generation",
        "gender",
        "category",
        "state",
        "market",
    ]
    kda_insights = _run_segment_analysis_tree_search(
        TARGET_MERCHANTS, target_col, dim_cols
    )
    print(f"\nKey Driver Analysis insights:\n{kda_insights}")

    response = llm.invoke(
        [
            SystemMessage(
                content=f"You are a senior analyst trying to analyze what drives the gap between peer and target performance.\n You run a tree search algorithm, and find the key drivers as:\n {kda_insights}\n Now, summarize the results in a clear and concise way."
            )
        ]
    )
    print("\n***LLM Summary of Key Driver Analysis:***")
    print(response.content)
