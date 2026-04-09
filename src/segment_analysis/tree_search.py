class TreeNode:
    def __init__(self, df, target_col, dimensions, max_depth=6):
        self.df = df
        self.target_col = target_col
        self.dimensions = dimensions
        self.max_depth = max_depth
        assert [
            d for d in dimensions if d in df.columns
        ], "All dimensions must be in the dataframe"
        self.left_node = None
        self.right_node = None

    def learn(self):
        if self.max_depth == 0 or len(self.dimensions) == 0:
            return

        # # find the best dimension to split on
        # best_dim = None
        # best_score = float("-inf")
        # for dim in self.dimensions:
        #     score = self.evaluate_split(dim)
        #     if score > best_score:
        #         best_score = score
        #         best_dim = dim

        # if best_dim is None:
        #     return

        # # split the dataframe on the best dimension
        # left_df = self.df[self.df[best_dim] == 0]
        # right_df = self.df[self.df[best_dim] == 1]

        # # create child nodes
        # remaining_dims = [d for d in self.dimensions if d != best_dim]
        # self.left_node = TreeNode(left_df, self.target_col, remaining_dims, self.max_depth - 1)
        # self.right_node = TreeNode(right_df, self.target_col, remaining_dims, self.max_depth - 1)

        # # recursively learn on child nodes
        # self.left_node.learn()
        # self.right_node.learn()
