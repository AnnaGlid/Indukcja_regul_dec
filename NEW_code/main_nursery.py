import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('algorithms/data/nursery_preprocessed.csv')
data_original = pd.read_csv('algorithms/data/nursery.csv')
original_attributes = data_original.columns

decision_class = 'class'
conditional = [col for col in data.columns if col != decision_class]
train, test = train_test_split(data, test_size=0.3)
X_train = train.drop(columns=[decision_class])
y_train = train[decision_class]
X_test = test.drop(columns=[decision_class])
y_test = test[decision_class]

trees_numbers = [5, 10, 15, 20]
class_dict = {
    0: 'not_recom',
    1: 'priority',
    2: 'spec_prior'
}
and_deli = ' and '
then_deli = ' then '
if_deli = 'if '
class_deli = 'class = '

results_depth_keys = [
    'trees_number', 'depth', 'depth_abs', 
    'avg_nodes_count', 'avg_tree_depth', 
    'alpha', 'min_rule_len', 'max_rule_len', 'avg_rule_len',
    'support', 'accuracy', 'recall', 'precision'
    # 'min_matching', 'max_matching', 'avg_matching'
]
results_depth = {key: [] for key in results_depth_keys}
results_imp_keys = [
    'trees_number', 'impurity_decrease', 
    'avg_nodes_count', 'avg_tree_depth', 
    'alpha', 'min_rule_len', 'max_rule_len', 'avg_rule_len',
    'support', 'accuracy', 'recall', 'precision'
    # 'min_matching', 'max_matching', 'avg_matching'
]
results_imp = {key: [] for key in results_imp_keys}

#region functions
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
            assert threshold == 0.5, f"Threshold not equal 0.5: {threshold}"
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
            rule = and_deli.join(path)
            paths.append(f"{if_deli}{rule}{then_deli}{class_deli}{class_label}")
    recurse(0, [])
    return paths

def get_all_rules_from_forest(forest):
    all_rules = []
    for tree in forest.estimators_:
        rules = extract_rules(tree, forest.feature_names_in_, tree.classes_)
        all_rules.extend(rules)
    return set(all_rules)

def get_descriptors(rule) -> set:    
    rule = rule.split(then_deli)[0]
    rule = rule.split(if_deli)[1]
    descriptors = rule.split(and_deli)
    return set([d for d in descriptors if d])

def get_decision(rule):
    return rule.split(then_deli)[1].split(class_deli)[1]

def get_variant(desc, original_attr):
    return desc.split('=')[0].replace(f'{original_attr}_', '')

def heuristic(all_rules_forest: set, decision: str, alpha: float) -> str:
    assert 0 <= alpha < 1
    first_descriptor = True
    h_rule = if_deli    

    i0 = set()
    for rule in all_rules_forest:
        if get_decision(rule) == decision:
            i0.add(rule)    
    ii = [r for r in i0.copy() if get_descriptors(r)]
    while len(i0.difference(ii)) < len(i0) * (1 - alpha):
        descriptors = set()
        for rule in ii:
            descriptors.update(get_descriptors(rule))
                        
        # remove descriptors that are in rule
        for desc in descriptors.copy():
            if desc in h_rule:
                descriptors.remove(desc)

        # choose descriptor which is the most popular in ii
        max_desc_num = 0
        max_desc = None
        for desc in descriptors:
            occurences = len(list(filter(lambda x: desc in x, ii)))
            if occurences > max_desc_num:
                max_desc_num = occurences
                max_desc = desc
        if first_descriptor: 
            h_rule += max_desc
            first_descriptor = False
        else:
            h_rule += f'{and_deli}{max_desc}'
            
        # update ii
        or_desc = next(filter(lambda x: max_desc.startswith(x), original_attributes))
        ii_rules_w_orig_desc = list(filter(lambda x: or_desc in x, ii))
        max_desc_attr, max_desc_val = max_desc.split('=')
        h_rule_descs = get_descriptors(h_rule)
        for ii_rule in ii_rules_w_orig_desc:
            # delete if subset
            ii_rule_descs = get_descriptors(ii_rule)
            if ii_rule_descs.issubset(h_rule_descs):
                ii.remove(ii_rule)
                continue

            # delete if incompatibile     
            delete = False       
            ii_descs = list(filter(lambda x: or_desc in x, ii_rule_descs))
            for ii_desc in ii_descs:
                ii_desc_attr, ii_desc_val = ii_desc.split('=')
                if max_desc_val == '1':
                    if (get_variant(max_desc_attr, or_desc) == get_variant(ii_desc_attr, or_desc) and ii_desc_val == '0') \
                        or (get_variant(max_desc_attr, or_desc) != get_variant(ii_desc_attr, or_desc) and ii_desc_val == '1'):
                        delete = True
                        break
                else:
                    if (get_variant(max_desc_attr, or_desc) == get_variant(ii_desc_attr, or_desc) and ii_desc_val == '1'):
                        delete = True
                        break
            if delete:
                ii.remove(ii_rule)
        d=1

    d=1
    h_rule += f'{then_deli}{class_deli}{decision}'
    return h_rule

def calculate_metrics(rule_set: set, table: pd.DataFrame) -> dict:
    confusion_matrix = {
        cl: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for cl in class_dict.values()
    }
    # support, accuracy, precision, recall, f1

    def covers(rule, row) -> bool:
        descs = get_descriptors(rule)
        for desc in descs:
            attr, val = desc.split('=')
            if row[attr] != float(val):
                return False
        return True
    covered = 0    
    for index, row in table.iterrows():
        for rule in rule_set:
            if covers(rule, row):
                rule_decision = get_decision(rule)
                true_decision = row[decision_class]
                if rule_decision == true_decision:
                    confusion_matrix[true_decision]['tp'] += 1
                    for dec in class_dict.values():
                        if dec != true_decision:
                            confusion_matrix[dec]['tn'] += 1
                else:
                    confusion_matrix[rule_decision]['fp'] += 1
                    confusion_matrix[true_decision]['fn'] += 1
                covered += 1
                continue


    # support - covered rows (without decision) to all the rows
    support = covered / len(table)

    # accuracy - correct predictions / all predictions
    accuracy = 0
    applies = 0
    for cl in class_dict.values():
        if sum(confusion_matrix[cl].values()) > 0:
            accuracy += (confusion_matrix[cl]['tp'] + confusion_matrix[cl]['tn']) / sum(confusion_matrix[cl].values())
            applies += 1
    if applies:
        accuracy = accuracy/ applies
    else:
        accuracy = 'NaN'

    # recall - correctly classified as cl / all cl. Avg of recall of all the classes
    recall = sum([
        confusion_matrix[cl]['tp'] / len(table.loc[table[decision_class]==cl]) for cl in class_dict.values()
    ]) / len(class_dict.values())

    # precision - correctly classified as cl / all classified as cl. Avg of precision of all the classes
    precision = 0
    applies = 0
    for cl in class_dict.values():
        if confusion_matrix[cl]['tp'] + confusion_matrix[cl]['fp'] != 0:
            precision += confusion_matrix[cl]['tp'] / (confusion_matrix[cl]['tp'] + confusion_matrix[cl]['fp'])
            applies += 1
    if applies:
        precision = precision / applies
    else:
        precision = 'NaN'
    
    return {
        'support': round(support, 4),
        'accuracy': round(accuracy, 4) if isinstance(accuracy, float) else accuracy,
        'recall': round(recall, 4),
        'precision': round(precision, 4) if isinstance(precision, float) else precision
    }


def get_results_for_forest(forest, all_rules_forest):
    results = {}
    trees_num = len(forest.estimators_)
    avg_nodes_count = round(sum(
        [estimator.tree_.node_count for estimator in forest.estimators_]
    ) / trees_num, 4)
    avg_tree_depth = round(sum(
        [estimator.tree_.max_depth for estimator in forest.estimators_]
    ) / trees_num, 4)    

    alpha_inc = 0.05    
    for alpha in np.arange(0, 0.2 + alpha_inc, alpha_inc):
        alpha = round(alpha, 2)
        rules = []
        for decision in class_dict.values():
            rules.append(heuristic(all_rules_forest, decision, alpha))
        rules = set(rules)
        rules_length = [rule.split(then_deli)[0].count('=') for rule in rules]
        results[alpha] = {
            'avg_nodes_count': avg_nodes_count,
            'avg_tree_depth': avg_tree_depth,
            'min_rule_len': min(rules_length),
            'max_rule_len': max(rules_length),
            'avg_rule_len': round(sum(rules_length) / len(rules_length), 4)
        }
        metrics = calculate_metrics(rules, test)
        results[alpha] = results[alpha] | metrics
    return results
#endregion        

for trees_number in trees_numbers:
    forest = RandomForestClassifier(n_estimators = trees_number)
    forest.fit(X_train, y_train)
    forest_max_depth = max([estimator.tree_.max_depth for estimator in forest.estimators_])
    for i in range(0, 5):
        print(f'Getting results trees number: {trees_number} and depth: max_depth - {i}')
        forest = RandomForestClassifier(n_estimators = trees_number, max_depth = forest_max_depth-i)
        forest.fit(X_train, y_train)
        all_rules_forest = get_all_rules_from_forest(forest)
        for alpha, forest_results in get_results_for_forest(forest, all_rules_forest).items():
            results_depth['trees_number'].append(trees_number)
            results_depth['depth'].append('max_depth' if not i else f'max_depth - {i}')
            results_depth['depth_abs'].append(forest_max_depth - i)
            results_depth['alpha'].append(alpha)
            for k, v in forest_results.items():
                results_depth[k].append(v)
            
    for imp_decrease in np.arange(0, 0.2, 0.05):
        imp_decrease = round(imp_decrease, 2)
        print(f'Getting results for {trees_number} and impurity decrease: {imp_decrease}')
        forest = RandomForestClassifier(n_estimators = trees_number, min_impurity_decrease = imp_decrease)
        forest.fit(X_train, y_train)
        all_rules_forest = get_all_rules_from_forest(forest)
        d=1
        for alpha, forest_results in get_results_for_forest(forest, all_rules_forest).items():
            results_imp['trees_number'].append(trees_number)
            results_imp['impurity_decrease'].append(imp_decrease)
            results_imp['alpha'].append(alpha)
            for k, v in forest_results.items():
                results_imp[k].append(v)
pd.DataFrame(results_depth).to_csv('results_depth.csv')
pd.DataFrame(results_imp).to_csv('results_imp.csv')
d=1



