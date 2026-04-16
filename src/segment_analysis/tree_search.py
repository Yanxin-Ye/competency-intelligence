import pandas as pd
from config import DIMENSION_COLS
import etl
from helper import grpby_dim_val


class TreeNode:
    def __init__(self, df: pd.DataFrame, target_col: str, dimensions: list[str]):
        """A node in the decision tree.
        Args:
            df[pd.DataFrame]: DataFrame for this node
            target_col[str]: The target column to search for
            dimensions[list[str]]: List of dimensions to consider for splitting
        """
        self.df = df.copy()
        assert (
            target_col in df.columns
        ), f"Target column '{target_col}' not found in DataFrame"
        self.target_col = target_col

        missing = [d for d in dimensions if d not in df.columns]
        assert not missing, f"Missing dimensions: {missing}"
        self.dimensions = [
            dim for dim in dimensions if df[dim].nunique(dropna=False) > 1
        ]  # Remove dimensions with only one unique value as they cannot be split further

        self.target_col_sum = self.df[self.target_col].sum()

        self.left_node = None
        self.right_node = None
        self.best_split_dim = None
        self.best_split_val = None
        self.best_score = None

    def learn(self, verbose=True):
        df = self.df

        if df.empty:
            return

        res = {}
        for dim in self.dimensions:
            res.update(grpby_dim_val(df, dim, self.target_col))

        if not res:
            return

        need_reverse = self.target_col_sum > 0
        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=need_reverse)
        if verbose:
            print(
                f"Top splits for current node with target sum {self.target_col_sum:.2%}:"
            )
            for i, (k, v) in enumerate(sorted_res[:10], 1):
                print(
                    f"{i:>2}. {k:<50} | {v:>8.2%} | contribution: {v/self.target_col_sum:>6.2%}"
                )

        best_split, best_score = sorted_res[0]
        best_split_dim, best_split_val = best_split.split(" = ")
        mask = df[best_split_dim].astype(str) == best_split_val

        left_df = df[mask]
        right_df = df[~mask | df[best_split_dim].isna()]

        # avoid useless split
        if left_df.empty or right_df.empty:
            return

        self.best_split_dim = best_split_dim
        self.best_split_val = best_split_val
        self.best_score = best_score

        self.left_node = TreeNode(left_df, self.target_col, self.dimensions)
        self.right_node = TreeNode(right_df, self.target_col, self.dimensions)

    def print_node(self):
        if self.best_split_dim is None:
            print(
                f"Leaf | total={self.df[self.target_col].sum():.2%} | {len(self.df)} rows"
            )
            return

        print(
            f"[{self.best_split_dim} == {self.best_split_val}] | "
            f"score={self.best_score:.2%} | "
            f"total={self.df[self.target_col].sum():.2%}"
        )
        print(
            f"  ├─ Left:  {self.left_node.df[self.target_col].sum():.2%} | "
            f"coverage={self.left_node.df[self.target_col].sum() / self.target_col_sum:.2%} | "
            f"{len(self.left_node.df)} rows"
        )
        print(
            f"  └─ Right: {self.right_node.df[self.target_col].sum():.2%} | "
            f"coverage={self.right_node.df[self.target_col].sum() / self.target_col_sum:.2%} | "
            f"{len(self.right_node.df)} rows"
        )


class Tree:
    def __init__(self, df, target_col, dimensions, max_coverage=0.2, max_depth=None):
        """A decision tree
        Args:
            df[pd.DataFrame]: DataFrame for the root node
            target_col[str]: The target column for the root node
            dimensions[list[str]]: List of dimensions to consider for splitting, also for the root node
            max_coverage[float]: Minimum coverage of the target column sum for a node to be split further, default to 0.2 (20%)
            max_depth[int]: Maximum depth of the tree

        """
        self.root = TreeNode(df, target_col, dimensions)
        self.target_col_sum = self.root.target_col_sum
        self.max_coverage = max_coverage
        self.max_depth = (
            min(max_depth, len(dimensions))
            if max_depth is not None
            else len(dimensions)
        )
        self.learned_outcomes = []  # To track learned outcomes and prevent duplicates

    def _learn_recursive(self, node: TreeNode, depth: int):
        if self.verbose:
            print(
                f"Current depth: {depth}, target sum at node: {node.df[node.target_col].sum():.2%}, rows: {len(node.df)}"
            )
        if node is None or depth >= self.max_depth:
            return

        node.learn(verbose=self.verbose)
        self.learned_outcomes.append(
            {
                "best_split_dim": node.best_split_dim,
                "best_split_val": node.best_split_val,
                "score": node.best_score,
                "coverage": node.df[node.target_col].sum() / self.target_col_sum,
            }
        )

        # stop if 1) this node could not split or 2) reached max depth - 1 (to ensure leaf nodes) or 3) coverage is too small
        if (
            node.left_node is None
            or node.right_node is None
            or depth == self.max_depth - 1
            or node.df[node.target_col].sum() / self.target_col_sum < self.max_coverage
        ):
            return

        self._learn_recursive(node.left_node, depth + 1)
        # self._learn_recursive(node.right_node, depth + 1)

    def learn(self, verbose=True):
        self.verbose = verbose
        self._learn_recursive(self.root, depth=0)

    def print_tree(self):
        self._print_tree(self.root, prefix="", side="Root", is_last=True)

    def _print_tree(self, node, prefix="", side="Root", is_last=True):
        connector = "└── " if is_last else "├── "

        if node is None:
            print(f"{prefix}{connector}{side}: None")
            return

        if node.best_split_dim is None:
            print(
                f"{prefix}{connector}{side}: Leaf | "
                f"parent={node.df[node.target_col].sum():.2%} | "
                f"coverage={node.df[node.target_col].sum() / self.target_col_sum :.2%}"
            )
            return

        print(
            f"{prefix}{connector}{side}: "
            f"[{node.best_split_dim} == {node.best_split_val}] | "
            f"score={node.best_score:.2%} | "
            f"parent={node.df[node.target_col].sum():.2%} | "
            f"coverage={node.df[node.target_col].sum() / self.target_col_sum :.2%}"
        )

        child_prefix = prefix + ("    " if is_last else "│   ")
        self._print_tree(node.left_node, child_prefix, "L", is_last=False)
        self._print_tree(node.right_node, child_prefix, "R", is_last=True)


if __name__ == "__main__":
    TARGET_MERCHANTS = "fraud_Kilback LLC"
    target_col = "amt_growth_ctc_diff"

    df_combined = etl.load_rca_data(TARGET_MERCHANTS)
    print("Data loaded successfully.")
    print(f"Data shape: {df_combined.shape}")
    print(f"Sample data:\n{df_combined.head()}")

    # node = TreeNode(df_combined, target_col, DIMENSION_COLS)
    # node.learn()
    # node.print_node()

    tree = Tree(df_combined, target_col, DIMENSION_COLS, max_coverage=0.25, max_depth=4)
    tree.learn(verbose=True)
    print("\n\nLearned tree structure:\n")
    tree.print_tree()
    print("\n\nLearned outcomes (to check for duplicates):")
    res = []
    for outcome in tree.learned_outcomes:
        res.append(f"{outcome['best_split_dim']} = {outcome['best_split_val']}")
    print("|".join(res))
