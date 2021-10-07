# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementation import *


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


def standardize(train_data, test_data):
    """Standardize the original data set."""
    mean = np.mean(train_data, 0)
    train_data = train_data - mean
    test_data = test_data - mean
    dev = np.std(train_data, 0)
    x = np.delete(train_data, np.where(dev == 0), axis=1)
    test_data = np.delete(test_data, np.where(dev == 0), axis=1)
    dev = np.delete(dev, np.where(dev == 0), axis=0)
    x = x / dev
    test_data = test_data / dev
    return x, test_data


def min_max_std(train_data, test_data):
    max_ = np.max(train_data, axis=0)
    min_ = np.min(train_data, axis=0)
    for i in range(train_data.shape[1]):
        train_data[:, i] = (train_data[:, i] - min_[i]) / (max_[i] - min_[i])
        test_data[:, i] = (test_data[:, i] - min_[i]) / (max_[i] - min_[i])

    return train_data, test_data


def delete_outlier(data, target):
    index = np.ones((len(data), 1), dtype=int)
    dev = np.std(data, axis=0)
    for i in range(len(dev)):
        index[np.where(data[:, i] > 3 * dev[i])] = 0

    data = np.delete(data, np.where(index == 0), axis=0)
    target = np.delete(target, np.where(index == 0), axis=0)
    return data, target


def compute_error(target, prediction):
    count = 0
    for i in range(len(target)):
        if target[i] != prediction[i]:
            count = count + 1
    return count


def build_poly(data, degree):
    a = np.ones((data.shape[0], 1))
    b = data
    for i in range(degree):
        a = np.hstack((a, b))
        for j in range(len(b)):
            b[j] = b[j] * data[j]
    return a


def pca(train_data, test_data, level=0.999):
    cov = np.cov(train_data, rowvar=0)
    eig_val, eig_vect = np.linalg.eig(cov)
    sorted_eig_val = np.sort(eig_val)[-1::-1]
    n = 0
    temp = 0
    sum = np.sum(eig_val)
    for i in sorted_eig_val:
        temp = temp + i
        n = n + 1
        if temp > level * sum:
            break

    eig_index = (np.argsort(eig_val))[-1:-(n + 1):-1]
    eig_vect = eig_vect[:, eig_index]
    train_data = train_data.dot(eig_vect)
    test_data = test_data.dot(eig_vect)
    return train_data, test_data


def convert_to_one_hot(x):
    y = np.zeros((len(x), 2))
    y[np.where(x == 1), 0] = 1
    y[np.where(x == -1), 1] = 1
    return y


def convert_to_original(x):
    y = np.zeros((len(x), 1))
    y[np.where(x[:, 0] == 1)] = 1
    y[np.where(x[:, 1] == 1)] = -1
    return y


def fix_empty(train_data, text_data, t=0.9):
    index = np.zeros((len(train_data), 1))
    i = 0
    while i < train_data.shape[1]:
        index[np.where(train_data[:, i] == -999)] = 1
        if np.sum(index) > t * len(train_data):
            train_data = np.delete(train_data, i, axis=1)
            text_data = np.delete(text_data, i, axis=1)
        else:
            value = np.median(train_data[np.where(train_data[:, i] != -999), i])
            train_data[np.where(train_data[:, i] == -999), i] = value
            text_data[np.where(text_data[:, i] == -999), i] = value
            i = i + 1
        index = np.zeros((len(train_data), 1))
    return train_data, text_data


def data_preprocess(x, y, test_data, poly=False, degree=2):
    xs = []
    ys = []
    ts = []
    index = x[:, 22]
    x = np.delete(x, 22, axis=1)
    test_index = test_data[:, 22]
    test_data = np.delete(test_index, 22, axis=1)
    for i in range(4):
        a, b = fix_empty(x[np.where(index == i)], test_data[np.where(test_index == i)])
        a, b = standardize(a, b)
        if poly:
            a = build_poly(a, degree)
            b = build_poly(b, degree)
        xs.append(a)
        ts.append(b)
        ys.append(y[np.where(index == i)])
    return xs, ys, ts


def cross_validation(x, y, n_fold=5, f=0, lambda_=0.0, epochs=300, gamma=1e-3, p=False, batch_size=50):
    a = 0
    step = int(len(x) / n_fold)
    result = 0
    while a < len(x):
        data = np.concatenate((x[:a], x[a + step:]), axis=0)
        targets = np.concatenate((y[:a], y[a + step:]), axis=0)
        test_data, test_targets = x[a:a + step], y[a:a + step]
        w = None
        if f == 1:
            loss, w = gradient_descent(targets, data, np.zeros((data.shape[1], targets.shape[1])), iteration=epochs,
                                       gamma=gamma, print_output=p)

        elif f == 2:
            loss, w = stochastic_gradient_descent(targets, data, np.zeros((data.shape[1], targets.shape[1])),
                                                  iteration=epochs,
                                                  gamma=gamma, batch_size=batch_size, print_output=p)

        elif f == 3:
            loss, w = least_squares(targets, data)

        elif f == 4:
            loss, w = ridge_regression(targets, data, lambda_)

        elif f == 5:
            loss, w = logistic_regression(targets, data,
                                          np.zeros((data.shape[1], targets.shape[1])), 10, 1e-6, False)

        elif f == 6:
            loss, w = reg_logistic_regression(targets, data,
                                              np.zeros((data.shape[1], targets.shape[1])), 10, 1e-6,
                                              lambda_=0.3,
                                              print_output=False)

        acc = compute_error(test_targets, predict_labels(w, test_data)) / step
        print(f, "train_error:", acc)
        result = result + acc
        a = a + step
    result = result / n_fold
    return result
