# pseudo-code showing how ensemable trees are used

def forest_predict(forest, input, is_regression=False):
    results = []

    for each_tree in forest:
        each_tree_result = each_tree.predict(input)
        results.append(each_tree_result)

    if is_regression:
        return average(results)
    else:
        return majority_vote(results)
