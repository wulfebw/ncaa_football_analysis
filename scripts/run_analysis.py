
import sys

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.dummy import DummyClassifier

import file_utils

DATA_FILENAME = 'data.csv'

##################################################
### Start Classification #########################
##################################################

def load_data_classification():
    filename = DATA_FILENAME
    data, labels = file_utils.read_csv_data_file(filename)
    data = np.array(data)
    samples = data[:, 3:-2]
    samples = preprocessing.scale(samples)
    targets_raw = data[:, -2:]
    targets = []
    for row in targets_raw:
        if row[0] > row[1]:
            targets.append(1)
        else:
            targets.append(0)

    percent = .8
    index = int(percent * len(samples))
    samples_train, targets_train = samples[:index], targets[:index]
    samples_test, targets_test = samples[index:], targets[index:]

    return samples_train, targets_train, samples_test, targets_test

def run_logistic_regression_analysis(samples_train, targets_train, samples_test, targets_test):
    """
    :test accuracy: 0.811764705882
    """
    lr = linear_model.LogisticRegression()
    lr.fit(samples_train, targets_train)
    accuracy = lr.score(samples_test, targets_test)
    print accuracy

def run_svm_classification_analysis(samples_train, targets_train, samples_test, targets_test):
    """
    :test accuracy: 0.811764705882
    """
    svm = SVC(kernel='rbf', C=1000, gamma=0.0001)
    svm.fit(samples_train, targets_train)
    accuracy = svm.score(samples_test, targets_test)
    print accuracy

def run_classification_baseline(samples_train, targets_train, samples_test, targets_test):
    """
    :test accuracy: 0.564705882353, 0.570588235294
    """
    svm = DummyClassifier(strategy='most_frequent',random_state=0)
    svm.fit(samples_train, targets_train)
    accuracy = svm.score(samples_test, targets_test)
    print accuracy

def plot_learning_curves(samples_train, targets_train, samples_test, targets_test):
    #model = SVC(kernel='rbf', C=1000, gamma=0.0001)
    model = linear_model.LogisticRegression()

    train_sizes = np.arange(.1, 1, .05)
    train_scores = []
    test_scores = []
    for train_size in train_sizes:
        index = int(train_size * len(samples_train))
        model.fit(samples_train[:index], targets_train[:index])
        train_score = model.score(samples_train[:index], targets_train[:index])
        train_scores.append(train_score)
        test_score = model.score(samples_test, targets_test)
        test_scores.append(test_score)

    plt.grid()
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score")
    plt.show()

##################################################
### Start Regression #############################
##################################################

def load_data_regression():
    filename = DATA_FILENAME
    data, labels = file_utils.read_csv_data_file(filename)
    data = np.array(data)
    samples = data[:, 3:-2]
    samples = preprocessing.scale(samples)
    targets_raw = data[:, -2]

    train_percentage = .8
    samples_train, targets_train = samples[:len(samples) * train_percentage], targets[:len(targets) * train_percentage]
    samples_test, targets_test = samples[len(samples) * train_percentage:], targets[len(targets) * train_percentage:]

    return samples_train, targets_train, samples_test, targets_test

def svm_grid_search(samples_train, targets_train, samples_test, targets_test):
    C_range = 10. ** np.arange(0, 5, 1)
    gamma_range = 10. ** np.arange(-6, -3, 1)
    param_grid = dict(gamma=gamma_range, C=C_range)

    grid = GridSearchCV(SVC(kernel='rbf', cache_size=10000), param_grid=param_grid)
    grid.fit(samples_train, targets_train)

    score_dict = grid.grid_scores_
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    pl.figure(figsize=(8, 6))
    pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
    pl.xlabel('gamma')
    pl.ylabel('C')
    pl.colorbar()
    pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    pl.yticks(np.arange(len(C_range)), C_range)
    pl.show()

def run_svm_regression_analysis(samples_train, targets_train, samples_test, targets_test):
    """
    :test error diff: 9.97
    """
    svr = SVR(kernel='rbf', C=25, gamma=0.0015)
    predictions = svr.fit(samples_train, targets_train).predict(samples_test)
    avg_diff = np.mean(abs(np.subtract(predictions, targets_test)))
    print avg_diff

def run_linear_regression_analysis(samples_train, targets_train, samples_test, targets_test):
    """
    :test error diff: 10.5
    """
    lr = linear_model.LinearRegression()
    lr.fit(samples_train, targets_train)
    predictions = lr.predict(samples_test)
    avg_diff = np.mean(abs(np.subtract(predictions, targets_test)))
    print avg_diff

if __name__ == '__main__':
    samples_train, targets_train, samples_test, targets_test = load_data_classification()
    plot_learning_curves(samples_train, targets_train, samples_test, targets_test)