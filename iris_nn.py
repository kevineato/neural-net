#!/usr/bin/env python

from nn import NeuralNetClf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def main():
    label = 'Species'
    train = pd.read_csv('iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv('iris_test.csv', names=CSV_COLUMN_NAMES, header=0)
    X, y = train.values, train.pop(label).values
    t_X, t_y = test.values, test.pop(label).values

    data_x = np.concatenate((X, t_X))
    data_y = np.concatenate((y, t_y))

    # For performing stratified k-fol cross-validation
    skf = StratifiedKFold(n_splits=10)

    k = 1
    counts = []
    for train, test in skf.split(data_x, data_y):
        train_x, test_x, train_y, test_y = data_x[train], data_x[test], data_y[train], data_y[test]

        # Initialize NN classifier
        nn = NeuralNetClf(n_hlayers=2, n_features=train_x.shape[1],
                          n_hidden=[5, 4], activation='relu', n_classes=len(SPECIES),
                          learning_rate=0.00005)

        # Train the NN
        nn.train(train_x, pd.get_dummies(train_y).values.astype(float), num_epochs=60000,
                 log_rate=10000)

        # Evaluate accuracy after training using separate test data
        accuracy, count = nn.eval(test_x, test_y)
        counts.append(count)
        print("\nEvaluation")
        print("*" * 50)
        print("Accuracy for k-fold split {}: {:.4f}\n".format(k, accuracy))
        k += 1

    # Print average accuracy from 10 rounds of stratified cross-validation
    print("\nFinal accuracy from 10-fold stratified cross-validation:")
    print("{:.4f}".format(np.sum(counts) / data_x.shape[0]))

if __name__ == '__main__':
    main()
