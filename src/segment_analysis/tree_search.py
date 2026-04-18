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
        self.best_split_score = None

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

        best_split, best_split_score = sorted_res[0]
        best_split_dim, best_split_val = best_split.split(" = ")
        mask = df[best_split_dim].astype(str) == best_split_val

        left_df = df[mask]
        right_df = df[~mask | df[best_split_dim].isna()]

        # avoid useless split
        if left_df.empty or right_df.empty:
            return

        self.best_split_dim = best_split_dim
        self.best_split_val = best_split_val
        self.best_split_score = best_split_score

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
            f"best_split_score={self.best_split_score:.2%} | "
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

    def _learn_recursive(self, node: TreeNode, depth: int):
        if self.verbose:
            print(
                f"Current depth: {depth}, target sum at node: {node.df[node.target_col].sum():.2%}, rows: {len(node.df)}"
            )
        if node is None or depth >= self.max_depth:
            return

        node.learn(verbose=self.verbose)

        # stop if 1) this node could not split or 2) reached max depth - 1 (to ensure leaf nodes) or 3) coverage is too small
        if (
            node.left_node is None
            or node.right_node is None
            or depth == self.max_depth - 1
            or node.left_node.target_col_sum / self.target_col_sum < self.max_coverage
        ):
            return

        self._learn_recursive(node.left_node, depth + 1)
        # self._learn_recursive(node.right_node, depth + 1)

    def learn(self, verbose=False):
        self.verbose = verbose
        self._learn_recursive(self.root, depth=0)

        self._collect_learned_outcomes(self.root)
        self._format_learned_outcomes()

    def _collect_learned_outcomes(self, node: TreeNode) -> list[dict]:
        """Collect the learned outcomes from the tree in a list of dicts.
        In this tree, left = best_split match, so leftmost path = most dominant segment chain
        """
        node = self.root
        path = []
        while node:
            path.append(
                {
                    "best_split_dim": node.best_split_dim,
                    "best_split_val": node.best_split_val,
                    "score": node.target_col_sum,
                    "coverage": node.target_col_sum / self.target_col_sum,
                }
            )
            if node.left_node is None:
                break
            node = node.left_node

        self._learned_outcomes = path

    def _format_learned_outcomes(self):
        assert hasattr(
            self, "_learned_outcomes"
        ), "Learned outcomes not collected yet. Call learn() first."
        order_map = {dim: i for i, dim in enumerate(DIMENSION_COLS)}
        sorted_outcomes = sorted(
            self._learned_outcomes,
            key=lambda o: order_map.get(o["best_split_dim"], float("inf")),
        )
        print("sorted outcomes: ", sorted_outcomes)
        key_segment = "|".join(
            f"{o['best_split_dim']} = {o['best_split_val']}"
            for o in sorted_outcomes
            if o["best_split_dim"]
        )
        self.key_segment = {
            "driver": key_segment,
            "best_score": self._learned_outcomes[-1]["score"],
            "coverage": self._learned_outcomes[-1]["coverage"],
        }

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
                f"score={node.target_col_sum:.2%} | "
                f"coverage={node.target_col_sum / self.target_col_sum :.2%}"
            )
            return

        print(
            f"{prefix}{connector}{side}: "
            # f"score={node.best_score:.2%} | "
            f"score={node.target_col_sum:.2%} | "
            f"coverage={node.target_col_sum / self.target_col_sum :.2%} | "
            f"[{node.best_split_dim} == {node.best_split_val}]"
        )

        child_prefix = prefix + ("    " if is_last else "│   ")
        self._print_tree(node.left_node, child_prefix, "L", is_last=False)
        self._print_tree(node.right_node, child_prefix, "R", is_last=True)


class TreeForest:
    def __init__(
        self, df, target_col, dimensions, n_trees=3, max_coverage=0.2, max_depth=None
    ):
        self.trees = []
        self.df = df
        self.target_col = target_col
        self.dimensions = dimensions
        self.n_trees = n_trees
        self.max_coverage = max_coverage
        self.max_depth = max_depth

    def construct_forest(self):
        # First Tree
        tree = Tree(
            self.df, self.target_col, self.dimensions, self.max_coverage, self.max_depth
        )
        tree.learn(verbose=False)
        self.trees.append(tree)


if __name__ == "__main__":
    TARGET_MERCHANTS = "fraud_Kilback LLC"
    target_col = "amt_growth_ctc_diff"

    df_combined = etl.load_rca_data(TARGET_MERCHANTS)
    print("Data loaded successfully.")
    print(f"Data shape: {df_combined.shape}")
    print(f"Sample data:\n{df_combined.head()}")

    ######## Test TreeNode ########
    # node = TreeNode(df_combined, target_col, DIMENSION_COLS)
    # node.learn()
    # node.print_node()

    ######## Test Tree ########
    tree = Tree(df_combined, target_col, DIMENSION_COLS, max_coverage=0.25, max_depth=4)
    tree.learn(verbose=True)
    print("\n\nLearned tree structure:\n")
    tree.print_tree()
    print("\nLearned outcomes: ")
    print(tree.key_segment)
