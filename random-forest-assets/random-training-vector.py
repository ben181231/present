# pseudo-code showing how random vectors are used in random forest building

import random

def random_forest_traning(data, number_of_trees = 10):
    trees = []

    for i in range(number_of_trees):
        training_vector = Vector(random.random())
        tree = Tree(training_vector)
        trees.append(tree)

    forest = build_forest(trees)
    forest.train(data)

    return forest
