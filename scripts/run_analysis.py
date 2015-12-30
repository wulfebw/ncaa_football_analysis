
import sys

import numpy as np
from sklearn import linear_model

import file_utils

DATA_FILENAME = 'data.csv'

def load_data():
    filename = DATA_FILENAME
    data, labels = file_utils.read_csv_data_file(filename)
    data = np.array(data)
    samples = data[:, 3:-2]
    targets = data[:, -2]

    samples_train, targets_train = samples[:len(samples) * .7], targets[:len(targets) * .7]
    samples_test, targets_test = samples[len(samples) * .7:], targets[len(targets) * .7:]

    return samples_train, targets_train, samples_test, targets_test

def run_svm_analysis(samples_train, targets_train, samples_test, targets_test):
    pass

def run_linear_regression_analysis(samples_train, targets_train, samples_test, targets_test):
    lr = linear_model.LinearRegression()
    lr.fit(samples_train, targets_train)
    predictions = lr.predict(samples_test)
    for target, pred in zip(targets_test, predictions):
        print 'target: {}'.format(target)
        print 'predicted: {}'.format(pred)
        print 'diff: {}'.format(target-pred)
        raw_input()

if __name__ == '__main__':
    
    samples_train, targets_train, samples_test, targets_test = load_data()
    run_linear_regression_analysis(samples_train, targets_train, samples_test, targets_test)