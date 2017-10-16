# pseudo-code showing how bagging is used for random tree training

def random_tree_training(tree, data):
    vector = tree.training_vector
    data_subset = get_data_subset(data, vector)
    tree.train(data_subset)
