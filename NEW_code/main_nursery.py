import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from sklearn.metrics import accuracy_score, classification_report

and_deli = ' and '
then_deli = ' then '
if_deli = 'if '
class_deli = 'class = '
trees_numbers = [5, 10, 15, 20, 25, 30]
REPETITION = 5


cwd = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(fr'{cwd}\data\nursery\nursery_preprocessed.csv')
data_original = pd.read_csv(fr'{cwd}\data\nursery\nursery.csv')
original_attributes = data_original.columns

results_depth_keys = [
    'trees_number', 'depth', 'depth_abs', 
    'avg_nodes_count', 'avg_tree_depth', 
    'alpha', 'min_rule_len', 'max_rule_len', 'avg_rule_len',
    'support', 'accuracy', 'recall', 'precision',
    'rules_number'
]
results_imp_keys = [
    'trees_number', 'impurity_decrease', 
    'avg_nodes_count', 'avg_tree_depth', 
    'alpha', 'min_rule_len', 'max_rule_len', 'avg_rule_len',
    'support', 'accuracy', 'recall', 'precision',
    'rules_number'
]
results_forest_keys =  [
    'trees_number', 'depth', 'depth_abs',
    'avg_nodes_count', 'avg_tree_depth', 
    'accuracy', 'recall', 'precision'
]
results_ir_keys = [
    'trees_number', 'depth', 'depth_abs', 
    'avg_nodes_count', 'avg_tree_depth', 
    'min_rule_len', 'max_rule_len', 'avg_rule_len',
    'support', 'accuracy', 'recall', 'precision',
    'rules_number'
]



decision_class = 'class'
conditional = [col for col in data.columns if col != decision_class]
train, test = train_test_split(data, test_size=0.3, stratify=data[decision_class])
X_train = train.drop(columns=[decision_class])
y_train = train[decision_class]
X_test = test.drop(columns=[decision_class])
y_test = test[decision_class]


class_dict = {
    0: 'not_recom',
    1: 'priority',
    2: 'spec_prior'
}

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
    i0 = set([r for r in i0.copy() if get_descriptors(r)])
    ii = set([r for r in i0.copy()])
    d=1
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
    h_rule += f'{then_deli}{class_deli}{decision}'
    return h_rule

def heuristic_v2(all_rules_forest: set, decision: str, alpha: float) -> list:
    assert 0 <= alpha < 1    
    rules_with_decision = set()
    for rule in all_rules_forest:
        if get_decision(rule) == decision:
            rules_with_decision.add(rule)        
    descriptors = set()
    for rule in rules_with_decision:
        descriptors.update(get_descriptors(rule))
    desc_occurences = {desc: len(list(filter(lambda x: desc in x, rules_with_decision))) for desc in descriptors}
    top_occurences = sorted(desc_occurences.values(), reverse=True)[:3]
    top_descs = [next(filter(lambda x: desc_occurences[x] == i, desc_occurences)) for i in top_occurences]

    rules = []
    for top_desc in top_descs:
        first_descriptor = True
        h_rule = if_deli    
        i0 = set([r for r in rules_with_decision.copy() if get_descriptors(r)])
        ii = set([r for r in i0.copy()])        
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
            if not first_descriptor:
                for desc in descriptors:
                    occurences = len(list(filter(lambda x: desc in x, ii)))
                    if occurences > max_desc_num:
                        max_desc_num = occurences
                        max_desc = desc
            if first_descriptor: 
                max_desc = top_desc
                occurences = len(list(filter(lambda x: max_desc in x, ii)))                
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
            for ii_rule in ii.copy():
                if max_desc not in ii_rule:
                    ii.remove(ii_rule)
        h_rule += f'{then_deli}{class_deli}{decision}'
        rules.append(h_rule)
    return rules


def calculate_metrics(rule_set: set, table: pd.DataFrame, most_common_decision: str) -> dict:
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
        row_is_covered = False
        for rule in rule_set:
            if covers(rule, row):
                row_is_covered = True
                rule_decision = get_decision(rule)
                true_decision =  row[decision_class]
                if rule_decision == true_decision:
                    confusion_matrix[true_decision]['tp'] += 1
                    for dec in class_dict.values():
                        if dec != true_decision:
                            confusion_matrix[dec]['tn'] += 1
                else:
                    confusion_matrix[rule_decision]['fp'] += 1
                    confusion_matrix[true_decision]['fn'] += 1
                    for dec in class_dict.values():
                        if dec not in [true_decision, rule_decision]:
                            confusion_matrix[dec]['tn'] += 1                    
                covered += 1
                break
        # None of the rules applied to this row
        if not row_is_covered:
            true_decision = row[decision_class]
            if most_common_decision == true_decision:
                confusion_matrix[true_decision]['tp'] += 1
                for dec in class_dict.values():
                    if dec != true_decision:
                        confusion_matrix[dec]['tn'] += 1
            else:
                confusion_matrix[most_common_decision]['fp'] += 1
                confusion_matrix[true_decision]['fn'] += 1
                for dec in class_dict.values():
                    if dec not in [true_decision, most_common_decision]:
                        confusion_matrix[dec]['tn'] += 1                    


    # support - covered rows (without decision) to all the rows
    support = covered / len(table)

    # accuracy - correct predictions (tp) / all predictions (all rows)
    correct_predictions = 0
    for cl in class_dict.values():
        correct_predictions += confusion_matrix[cl]['tp']
    accuracy = correct_predictions / len(table)

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

def get_results_for_forest(forest, all_rules_forest, most_common_decision: str, heu: str = 'v1'):
    results = {}
    trees_num = len(forest.estimators_)
    avg_nodes_count = round(sum(
        [estimator.tree_.node_count for estimator in forest.estimators_]
    ) / trees_num, 4)
    avg_tree_depth = round(sum(
        [estimator.tree_.max_depth for estimator in forest.estimators_]
    ) / trees_num, 4)    

    alpha_inc = 0.05 
    max_alpha = 0.2 if heu == 'v1' else 0.5
    for alpha in np.arange(0, max_alpha + alpha_inc, alpha_inc):
        alpha = round(alpha, 2)
        rules = []
        for decision in class_dict.values():
            if heu == 'v1':
                rules.append(heuristic(all_rules_forest, decision, alpha))
            if heu == 'v2':
                rules.extend(heuristic_v2(all_rules_forest, decision, alpha))
        rules = set(rules)
        rules_length = [rule.split(then_deli)[0].count('=') for rule in rules]
        results[alpha] = {
            'avg_nodes_count': avg_nodes_count,
            'avg_tree_depth': avg_tree_depth,
            'min_rule_len': min(rules_length) if rules_length else None,
            'max_rule_len': max(rules_length) if rules_length else None,
            'avg_rule_len': round(sum(rules_length) / len(rules_length), 4) if rules_length else None,
            'rules_number': len(rules)
        }
        metrics = calculate_metrics(rules, test, most_common_decision)
        results[alpha] = results[alpha] | metrics
    return results

def save_results(results_rep: list[dict], filename: str):
    results = {}
    for col, values in results_rep[0].items():
        if col in ['trees_number', 'depth', 'alpha']:
            results[col] = values
        elif 'min_' in col:
            results[col] = []
            for idx_val, val in enumerate(values):
                results[col].append(min([results_rep[idx_rep][col][idx_val] for idx_rep in range(REPETITION)]))
        elif 'max_' in col:
            results[col] = []
            for idx_val, val in enumerate(values):
                results[col].append(max([results_rep[idx_rep][col][idx_val] for idx_rep in range(REPETITION)]))
        else:        
            if col in ['support', 'accuracy', 'precision', 'recall']:
                new_col = 'avg_' + col
                results['min_'+col] = []
                results['max_'+col] = []
                results['std_'+col] = []
                for idx_val, val in enumerate(values):
                    vals = [results_rep[idx_rep][col][idx_val] for idx_rep in range(REPETITION)]
                    results['min_'+col].append(min(vals))
                    results['max_'+col].append(max(vals))
                    results['std_'+col].append(np.std(vals))
            else:
                new_col = col
            results[new_col] = []
            for idx_val, val in enumerate(values):            
                results[new_col].append(sum([results_rep[idx_rep][col][idx_val]
                                        for idx_rep in range(REPETITION)]) / REPETITION)  
    pd.DataFrame(results).to_csv(f'{filename}.csv')  
#endregion        

class_occurence = {val: list(y_train).count(val) for val in set(y_train.values)}
max_occurence = max(class_occurence.values())
most_common_decision = next(filter(lambda x: class_occurence[x] == max_occurence, class_occurence))
    

forest = RandomForestClassifier(n_estimators = trees_numbers[0])
forest.fit(X_train, y_train)
forest_max_depth = max([estimator.tree_.max_depth for estimator in forest.estimators_])
min_required_depth = math.ceil(math.log(len(class_dict), 2))

if False:
    #region expreriments: max tree depth
    results_depth_rep = []
    for repeat in range(REPETITION):
        results_depth_i = {key: [] for key in results_depth_keys}
        for trees_number in trees_numbers:
            for depth_diff in range(0, forest_max_depth - min_required_depth + 1):        
                print(f'Getting results: rep {repeat}, trees number: {trees_number} and depth: max_depth - {depth_diff}')
                forest = RandomForestClassifier(n_estimators = trees_number, max_depth = forest_max_depth-depth_diff)
                forest.fit(X_train, y_train)
                all_rules_forest = get_all_rules_from_forest(forest)
                for alpha, forest_results in get_results_for_forest(forest, all_rules_forest, most_common_decision).items():
                    results_depth_i['trees_number'].append(trees_number)
                    results_depth_i['depth'].append('max_depth' if not depth_diff else f'max_depth - {depth_diff}')
                    results_depth_i['depth_abs'].append(forest_max_depth - depth_diff)
                    results_depth_i['alpha'].append(alpha)
                    for k, v in forest_results.items():
                        results_depth_i[k].append(v)
        results_depth_rep.append(results_depth_i)
    save_results(results_depth_rep, 'results_depth')
    #endregion

if False:
    #region expreriments: impurity decrease
    results_imp_rep = []
    for repeat in range(REPETITION):
        results_imp_i = {key: [] for key in results_imp_keys}           
        for trees_number in trees_numbers:        
            for imp_decrease in [0, 0.001,  0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
                print(f'Getting results: rep {repeat}, for {trees_number} and impurity decrease: {imp_decrease}')
                forest = RandomForestClassifier(n_estimators = trees_number, min_impurity_decrease = imp_decrease)
                forest.fit(X_train, y_train)
                all_rules_forest = get_all_rules_from_forest(forest)
                d=1
                for alpha, forest_results in get_results_for_forest(forest, all_rules_forest, most_common_decision).items():
                    results_imp_i['trees_number'].append(trees_number)
                    results_imp_i['impurity_decrease'].append(imp_decrease)
                    results_imp_i['alpha'].append(alpha)
                    for k, v in forest_results.items():
                        results_imp_i[k].append(v)
        results_imp_rep.append(results_imp_i)
    save_results(results_imp_rep, 'results_imp')
    #endregion

if True:
    #region expreriments: random forest
    results_forest_rep = []
    for repeat in range(REPETITION):
        results_forest_i = {key: [] for key in results_forest_keys}
        for trees_number in trees_numbers:
            for depth_diff in range(0, forest_max_depth - min_required_depth + 1):        
                print(f'Getting results: rep {repeat}, trees number: {trees_number} and depth: {depth_diff}- RandomForest')
                forest = RandomForestClassifier(n_estimators = trees_number)
                forest.fit(X_train, y_train)           
                y_pred_test = forest.predict(X_test)       
                class_report = classification_report(y_test, y_pred_test, output_dict=True)    
                trees_num = len(forest.estimators_)
                avg_nodes_count = round(sum(
                    [estimator.tree_.node_count for estimator in forest.estimators_]
                ) / trees_num, 4)
                avg_tree_depth = round(sum(
                    [estimator.tree_.max_depth for estimator in forest.estimators_]
                ) / trees_num, 4)           
                results_forest_i['trees_number'].append(trees_number)
                results_forest_i['depth'].append('max_depth' if not depth_diff else f'max_depth - {depth_diff}')
                results_forest_i['depth_abs'].append(forest_max_depth - depth_diff)                
                results_forest_i['avg_nodes_count'].append(avg_nodes_count)
                results_forest_i['avg_tree_depth'].append(avg_tree_depth)
                results_forest_i['accuracy'].append(class_report['accuracy'])
                results_forest_i['recall'].append(class_report['macro avg']['recall'])
                results_forest_i['precision'].append(class_report['macro avg']['precision'])
        results_forest_rep.append(results_forest_i)
    save_results(results_forest_rep, 'results_forest')
    #endregion

if False:
    #region expreriments: inner rules
    results_ir_rep = []
    for repeat in range(REPETITION):
        results_ir_i = {key: [] for key in results_ir_keys}
        for trees_number in trees_numbers:
            for depth_diff in range(0, forest_max_depth - min_required_depth + 1):        
                print(f'Getting results: rep {repeat}, trees number: {trees_number} and depth: max_depth - {depth_diff}: Inner Rules')
                forest = RandomForestClassifier(n_estimators = trees_number, max_depth = forest_max_depth-depth_diff)
                forest.fit(X_train, y_train)
                all_rules_forest = get_all_rules_from_forest(forest)
                trees_num = len(forest.estimators_)
                avg_nodes_count = round(sum(
                    [estimator.tree_.node_count for estimator in forest.estimators_]
                ) / trees_num, 4)
                avg_tree_depth = round(sum(
                    [estimator.tree_.max_depth for estimator in forest.estimators_]
                ) / trees_num, 4)    

                rules = set(all_rules_forest)
                rules_length = [rule.split(then_deli)[0].count('=') for rule in rules]
                results_ir_i['trees_number'].append(trees_num)
                results_ir_i['depth'].append('max_depth' if not depth_diff else f'max_depth - {depth_diff}')
                results_ir_i['depth_abs'].append(forest_max_depth - depth_diff)                
                results_ir_i['avg_nodes_count'].append(avg_nodes_count)
                results_ir_i['avg_tree_depth'].append(avg_tree_depth)
                results_ir_i['min_rule_len'].append(min(rules_length))
                results_ir_i['max_rule_len'].append(max(rules_length))
                results_ir_i['avg_rule_len'].append(round(sum(rules_length) / len(rules_length), 4))
                results_ir_i['rules_number'].append(len(rules))                
                metrics = calculate_metrics(rules, test, most_common_decision)
                results_ir_i['support'].append(metrics['support'])
                results_ir_i['accuracy'].append(metrics['accuracy'])
                results_ir_i['recall'].append(metrics['recall'])
                results_ir_i['precision'].append(metrics['precision'])         
        results_ir_rep.append(results_ir_i)    
    save_results(results_ir_rep, 'results_inner_rules')
    #endregion

if False:
    #region expreriments: depth heuristic v2
    results_depth_rep = []
    for repeat in range(REPETITION):
        results_depth_i = {key: [] for key in results_depth_keys}
        for trees_number in trees_numbers:
            for depth_diff in range(0, forest_max_depth - min_required_depth + 1):        
                print(f'Getting results: rep {repeat}, hv2 trees number: {trees_number} and depth: max_depth - {depth_diff}')
                forest = RandomForestClassifier(n_estimators = trees_number, max_depth = forest_max_depth-depth_diff)
                forest.fit(X_train, y_train)
                all_rules_forest = get_all_rules_from_forest(forest)
                for alpha, forest_results in get_results_for_forest(forest, all_rules_forest, most_common_decision, 'v2').items():
                    results_depth_i['trees_number'].append(trees_number)
                    results_depth_i['depth'].append('max_depth' if not depth_diff else f'max_depth - {depth_diff}')
                    results_depth_i['depth_abs'].append(forest_max_depth - depth_diff)
                    results_depth_i['alpha'].append(alpha)
                    for k, v in forest_results.items():
                        results_depth_i[k].append(v)
        results_depth_rep.append(results_depth_i)
    save_results(results_depth_rep, 'results_hv2_depth')
    #endregion

if False:
    #region expreriments: impurity decrease heuristic v2: 
    results_imp_rep = []
    for repeat in range(REPETITION):
        results_imp_i = {key: [] for key in results_imp_keys}           
        for trees_number in trees_numbers:        
            for imp_decrease in [0, 0.001,  0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02]:
                print(f'Getting results: rep {repeat}, for {trees_number} and impurity decrease: {imp_decrease}')
                forest = RandomForestClassifier(n_estimators = trees_number, min_impurity_decrease = imp_decrease)
                forest.fit(X_train, y_train)
                all_rules_forest = get_all_rules_from_forest(forest)
                d=1
                for alpha, forest_results in get_results_for_forest(forest, all_rules_forest, most_common_decision, 'v2').items():
                    results_imp_i['trees_number'].append(trees_number)
                    results_imp_i['impurity_decrease'].append(imp_decrease)
                    results_imp_i['alpha'].append(alpha)
                    for k, v in forest_results.items():
                        results_imp_i[k].append(v)
        results_imp_rep.append(results_imp_i)
    save_results(results_imp_rep, 'results_v2_imp')
    #endregion
print('Koniec')
