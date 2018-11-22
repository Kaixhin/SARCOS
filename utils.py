import numpy as np


def get_tree_param_counts(estimator):
  n_nodes = estimator.tree_.node_count
  left_children = list(estimator.tree_.children_left)
  right_children = list(estimator.tree_.children_right)
  
  node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
  is_leaf = np.zeros(shape=n_nodes, dtype=bool)
  stack = [(0, -1)]  # Seed is the root note ID and its parent depth
  while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (left_children[node_id] != right_children[node_id]):
      stack.append((left_children[node_id], parent_depth + 1))
      stack.append((right_children[node_id], parent_depth + 1))
    else:
      is_leaf[node_id] = True

  params = n_nodes + np.sum(is_leaf) * (estimator.n_outputs_ - 1)  # Add extra params for multivariate ouputs
  node_depth = node_depth[is_leaf]
  path_params = node_depth + estimator.n_outputs_ - 1  # Add extra params for multivariate ouputs
  return params, np.mean(path_params)
