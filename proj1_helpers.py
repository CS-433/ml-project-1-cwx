# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementation import *
from util import *


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def one_hot_predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.zeros((len(data), 1))
    t = np.dot(data, weights)
    y_pred[np.where(t[:, 0] >= t[:, 1])] = 1
    y_pred[np.where(t[:, 0] <= t[:, 1])] = -1
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def standardize(train_data, test_data=None):
    """Standardize the original data set, deleting column with standard deviation 0."""
    mean = np.mean(train_data, 0)
    train_data = train_data - mean
    if test_data is not None:
        test_data = test_data - mean
    dev = np.std(train_data, 0)
    train_data = np.delete(train_data, np.where(dev == 0), axis=1)
    if test_data is not None:
        test_data = np.delete(test_data, np.where(dev == 0), axis=1)
    dev = np.delete(dev, np.where(dev == 0), axis=0)
    train_data = train_data / dev
    if test_data is not None:
        test_data = test_data / dev
    return train_data, test_data


def lognormal(train_data, test_data=None):
    """Standardize the log transformed data set ."""
    train_data = train_data + 1000
    train_data = np.log(train_data)
    if test_data is not None:
        test_data = test_data + 1000
        test_data = np.log(test_data)
    train_data, test_data = standardize(train_data, test_data)
    return train_data, test_data


def min_max_std(train_data, test_data):
    """Map the data value to the interval from 0 to 1."""
    max_ = np.max(train_data, axis=0)
    min_ = np.min(train_data, axis=0)
    train_data = (train_data - min_) / (max_ - min_)
    test_data = (test_data - min_) / (max_ - min_)
    return train_data, test_data


def delete_outlier(data, target, t=3):
    """Delete extreme value."""
    dev = np.std(data, axis=0)
    for i in range(data.shape[1]):
        target = target[np.abs(data[:, i]) < t * dev[i]]
        data = data[np.abs(data[:, i]) < t * dev[i]]
    return data, target


def accuracy(target, prediction):
    """Return the accuracy of the prediction."""
    return np.mean(target == prediction)


def build_poly(data, degree):
    """Generate polynomial feature without cross terms."""
    output = np.hstack((data,
                        *[data ** d for d in range(2, degree + 1)]))
    output = (output - np.mean(output, axis=0)) / np.std(output, axis=0)
    return np.hstack((np.ones((data.shape[0], 1)), output))


def build_poly2(data, degree):
    """Generate polynomial feature with cross terms."""
    coef = np.zeros((degree, data.shape[1]))
    coef[0] = 1
    for i in range(1, degree):
        for j in range(data.shape[1] - 1, -1, -1):
            coef[i, j] = np.sum(coef[i - 1, j:])
    temp2 = data.astype('float64')
    out = data.astype('float64')
    for i in range(1, degree):
        temp1 = temp2
        for index, j in enumerate(coef[i]):
            if index == 0:
                temp2 = np.array(
                    [temp1[:, l] * data[:, index] for l in range(int(temp1.shape[1] - j), temp1.shape[1])],
                    dtype='float64').transpose()
            else:
                temp2 = np.hstack((temp2, np.array([temp1[:, l] * data[:, index] for l in
                                                    range(int(temp1.shape[1] - j), temp1.shape[1])],
                                                   dtype='float64').transpose()))
        out = np.hstack((out, temp2)).astype('float64')
    # out = (out - np.mean(out, axis=0)) / np.std(out, axis=0)
    return np.hstack((np.ones((data.shape[0], 1), dtype='float64'), out)).astype('float64')


def pca(train_data, test_data, level=0.99):
    """Project the data into a lower dimension"""
    cov = np.cov(train_data, rowvar=0)
    eig_val, eig_vect = np.linalg.eig(cov)
    sorted_eig_val = np.sort(eig_val)[-1::-1]
    n = 0
    temp = 0
    for i in sorted_eig_val:
        temp = temp + i
        n = n + 1
        if temp > level * np.sum(eig_val):
            break

    eig_index = (np.argsort(eig_val))[-1:-(n + 1):-1]
    eig_vect = eig_vect[:, eig_index]
    train_data = train_data.dot(eig_vect)
    test_data = test_data.dot(eig_vect)
    return train_data, test_data


def convert_to_one_hot(x):
    """Use one hot encoding"""
    y = np.zeros((len(x), 2))
    y[np.where(x == 1), 0] = 1
    y[np.where(x == -1), 1] = 1
    return y


def convert_to_original(x):
    """Convert one hot data in to original data"""
    y = np.zeros((len(x), 1))
    y[np.where(x[:, 0] == 1)] = 1
    y[np.where(x[:, 1] == 1)] = -1
    return y


def fix_empty(train_data, text_data, t=0.9):
    """Fix missing value,missing value will be replaced with the median.
    Column with too many missing values will be deleted."""
    index = np.zeros((len(train_data), 1))
    i = 0
    while i < train_data.shape[1]:
        index[np.where(train_data[:, i] == -999)] = 1
        if np.sum(index) > t * len(train_data):
            train_data = np.delete(train_data, i, axis=1)
            text_data = np.delete(text_data, i, axis=1)
        else:
            value = np.nanmedian(train_data[np.where(index == 0), i])
            train_data[np.where(index == 1), i] = value
            text_data[np.where(text_data[:, i] == -999), i] = value
            i = i + 1
        index = np.zeros((len(train_data), 1))
    return train_data, text_data


def split_data_by_categories(x, y, test_data, test_id):
    """Split data into 4 group by category number."""
    xs = []
    ys = []
    ts = []
    ids = []
    index = x[:, 22]
    test_index = test_data[:, 22]
    x = np.delete(x, 22, axis=1)
    test_data = np.delete(test_data, 22, axis=1)
    for i in range(4):
        xs.append(x[np.where(index == i)])
        ys.append(y[np.where(index == i)])
        ts.append(test_data[np.where(test_index == i)])
        ids.append(test_id[np.where(test_index == i)])
    return xs, ys, ts, ids


def preprocess(train_data, train_target, test_data, ids, split=True, std_=False, type='z-norm', poly=False,
               cross_term=False, degree=1, fix_emp=False, embedding=False, t=0.99):
    """Preprocess the data"""
    if split:

        xs, ys, ts, ids = split_data_by_categories(train_data, train_target, test_data, ids)
    else:
        xs, ys, ts, ids = [train_data], [train_target], [test_data], [ids]

    if fix_emp:
        for i in range(len(xs)):
            xs[i], ts[i] = fix_empty(xs[i], ts[i])

    if std_:
        if type == 'z-norm':
            for i in range(len(xs)):
                xs[i], ts[i] = standardize(xs[i], ts[i])
        elif type == 'log-norm':
            for i in range(len(xs)):
                xs[i], ts[i] = lognormal(xs[i], ts[i])
        else:
            for i in range(len(xs)):
                xs[i], ts[i] = min_max_std(xs[i], ts[i])

    if embedding:
        for i in range(len(xs)):
            xs[i], ts[i] = pca(xs[i], ts[i], t)

    if poly:
        if cross_term:
            xs = [build_poly2(xs[i], degree) for i in range(len(xs))]
            ts = [build_poly2(ts[i], degree) for i in range(len(ts))]
        else:
            xs = [build_poly(xs[i], degree) for i in range(len(xs))]
            ts = [build_poly(ts[i], degree) for i in range(len(ts))]

    return xs, ys, ts, ids


def cross_validation(x, y, n_fold=5, f='gradient_descent', lambda_=0.0, epochs=300, gamma=1e-1, batch_size=50,
                     n_units=128, n_layers=3, print_res=False,
                     delete_out=False, level=3, fix_emp=False, t=0.9, normalize=False, norm_type='z-norm', poly=False,
                     cross_term=False, degree=1, embedding=False, embedding_level=0.99):
    """Cross validation"""
    a = 0
    step = int(0.9 + len(x) / n_fold)
    result = 0
    count = 0
    while a < len(x):
        end = a + step
        if end > len(x):
            end = len(x)
        data = np.concatenate((x[:a], x[end:]), axis=0)
        targets = np.concatenate((y[:a], y[end:]), axis=0)
        test_data, test_targets = x[a:end], y[a:end]
        if fix_emp:
            data, test_data = fix_empty(data, test_data, t)
        if normalize:
            if norm_type == 'z-norm':
                data, test_data = standardize(data, test_data)
            elif norm_type == 'log-norm':
                data, test_data = lognormal(data, test_data)
            else:
                data, test_data = min_max_std(data, test_data)
        if delete_out:
            data, targets = delete_outlier(data, targets, level)
        if embedding:
            data, test_data = pca(data, test_data, embedding_level)
        if poly:
            if cross_term:
                data = build_poly2(data, degree)
                test_data = build_poly2(test_data, degree)
            else:
                data = build_poly(data, degree)
                test_data = build_poly(test_data, degree)

        w = None
        if f == 'gradient_descent':
            loss, w = gradient_descent(targets, data, np.zeros((data.shape[1], targets.shape[1])), iteration=epochs,
                                       gamma=gamma, print_output=print_res)

        elif f == 'stochastic_gradient_descent':
            loss, w = stochastic_gradient_descent(targets, data, np.zeros((data.shape[1], targets.shape[1])),
                                                  iteration=epochs, gamma=gamma, batch_size=batch_size,
                                                  print_output=print_res)

        elif f == 'least_squares':
            loss, w = least_squares(targets, data)

        elif f == 'ridge_regression':

            loss, w = ridge_regression(targets, data, lambda_)

        elif f == 'logistic_regression':
            loss, w = logistic_regression(targets, data,
                                          np.zeros((data.shape[1], targets.shape[1])), iteration=epochs, gamma=gamma,
                                          print_output=print_res)
        elif f == 'reg_logistic_regression':
            loss, w = reg_logistic_regression(targets, data,
                                              np.zeros((data.shape[1], targets.shape[1])), iteration=epochs,
                                              gamma=gamma, print_output=print_res, lambda_=lambda_)
        if f == 'dnn':
            net = Sequential(Linear(data.shape[1], n_units), ReLU())
            for i in range(n_layers - 2):
                net.add_module(Linear(n_units, n_units))
                net.add_module(ReLU())
            net.add_module(Linear(n_units, targets.shape[1]))
            train_model(data, targets, net, print_res=print_res, nb_epochs=epochs, lambda_l2=lambda_,
                        learning_rate=gamma, test_data=test_data, test_target=test_targets, mini_batch_size=batch_size)
            output = net.forward(test_data)
            predicted = np.ones((len(output), 1), dtype=int)
            predicted[np.where(output[:, 0] > output[:, 1])] = 0
            acc = 1 - compute_error(test_targets, predicted) / test_data.shape[0]
        else:
            acc = accuracy(test_targets, predict_labels(w, test_data))
        print(f, "accuracy:", acc)
        result = result + acc
        count += 1
        a = a + step
    result = result / count
    return result
