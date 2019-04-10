#!/usr/bin/env python

from nn import NeuralNetClf
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

def main():
    # Load data from sklearn
    cancer_data = load_breast_cancer()

    data_x = cancer_data.data
    data_y = cancer_data.target

    # Normalize inputs
    data_x = normalize(data_x)

    skf = StratifiedKFold(n_splits=10)

    k = 1
    counts = []
    for train, test in skf.split(data_x, data_y):
        train_x, test_x, train_y, test_y = data_x[train], data_x[test], data_y[train], data_y[test]

        # Initialize network
        nn = NeuralNetClf(n_hlayers=2, n_features=train_x.shape[1], n_hidden=[10, 8],
                          activation='tanh', n_classes=2, learning_rate=0.0002)

        # Train network
        nn.train(train_x, pd.get_dummies(train_y).values.astype(float),
                 num_epochs=20000, log_rate=4000)

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
