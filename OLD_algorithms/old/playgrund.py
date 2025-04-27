import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


data = pd.read_csv('nursery_preprocessed.csv')
decision_class = 'class'
conditional = [col for col in data.columns if col != decision_class]
train, test = train_test_split(data, test_size=0.3)
X_train = train.drop(columns=[decision_class])
y_train = train[decision_class]
X_test = test.drop(columns=[decision_class])


y_test = test[decision_class]

forest = RandomForestClassifier(n_estimators = 1, max_depth=3, random_state=5)
forest.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

tree = forest.estimators_[0]

class_dict = {
    0: 'not_recom',
    1: 'priority',
    2: 'spec_prior'
}

def extract_rules(tree, feature_names, class_names=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!" for i in tree_.feature
    ]

    paths = []

    def recurse(node, path):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # left
            # recurse(tree_.children_left[node], path + [f"({name} <= {threshold:.4f})"])
            recurse(tree_.children_left[node], path + [f"{name}=0"])
            # right
            # recurse(tree_.children_right[node], path + [f"({name} > {threshold:.4f})"])
            recurse(tree_.children_right[node], path + [f"{name}=1"])
        else:
            value = tree_.value[node][0]
            class_id = np.argmax(value)
            class_label = class_dict[class_names[class_id]] if class_names is not None else class_id
            rule = " and ".join(path)
            paths.append(f"if {rule} then class = {class_label}")

    recurse(0, [])
    return paths

rules = extract_rules(tree, forest.feature_names_in_, tree.classes_)

for r in rules:
    print(r)

plot_tree(tree, feature_names=conditional, class_names=sorted(set(data['class'].values)))
plt.show()


#region analisys
# n_nodes = tree.tree_.node_count
# children_left = tree.tree_.children_left
# children_right = tree.tree_.children_right
# feature = tree.tree_.feature
# threshold = tree.tree_.threshold
# values = tree.tree_.value

# node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# stack = [(0, 0)]  # seed is the root node id and its parent depth
# while len(stack) > 0:
#     node_id, parent_depth = stack.pop()
#     node_depth[node_id] = parent_depth + 1

#     # If we have a test node
#     if (children_left[node_id] != children_right[node_id]):
#         stack.append((children_left[node_id], parent_depth + 1))
#         stack.append((children_right[node_id], parent_depth + 1))
#     else:
#         is_leaves[node_id] = True   

# print(f"The binary tree structure has {n_nodes} nodes and has the following tree structure:")
# for i in range(n_nodes):
#     if is_leaves[i]:
#         predicted_class = np.argmax(values[i])
#         class_label = class_dict[tree.classes_[predicted_class]]
#         print(f"{node_depth[i] * "\t"} node={i} leaf node with class {class_label}.")
#     else:
#         # print(f"{node_depth[i] * "\t"}node={i} test node: go to node {children_left[i]} if X[:, {feature[i]}] <= {threshold[i]} else to {children_right[i]} node %s.")
#         print(f"{node_depth[i] * "\t"}node={i} test node: go to node {children_left[i]} if {forest.feature_names_in_[feature[i]]} <= {threshold[i]} else to node {children_right[i]}.")
# print()
#endregion



d=1