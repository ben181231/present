# pseudo-code showing how random split selection is done in tree training

def random_split_selection(tree, working_subtree, data):
    vector = tree.training_vector
    features = get_feature_subset(data.features, vector)
    best_split = working_subtree.best_split(features, data)
    tree.add_split(working_subtree, best_split)
