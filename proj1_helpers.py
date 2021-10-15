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


def standardize(train_data, test_data):
    """Standardize the original data set."""
    mean = np.mean(train_data, 0)
    train_data = train_data - mean
    test_data = test_data - mean
    dev = np.std(train_data, 0)
    train_data = np.delete(train_data, np.where(dev == 0), axis=1)
    test_data = np.delete(test_data, np.where(dev == 0), axis=1)
    dev = np.delete(dev, np.where(dev == 0), axis=0)
    train_data = train_data / dev
    test_data = test_data / dev
    return train_data, test_data


def min_max_std(train_data, test_data):
    max_ = np.max(train_data, axis=0)
    min_ = np.min(train_data, axis=0)
    train_data = (train_data - min_) / (max_ - min_)
    test_data = (test_data - min_) / (max_ - min_)
    return train_data, test_data


def delete_outlier(data, target, t=3):
    dev = np.std(data, axis=0)
    for i in range(data.shape[1]):
        target = target[np.abs(data[:, i]) < t * dev[i]]
        data = data[np.abs(data[:, i]) < t * dev[i]]
    return data, target


def accuracy(target, prediction):
    return np.mean(target == prediction)


def build_poly(data, degree):
    output = np.hstack((data,
                        *[data ** d for d in range(2, degree + 1)]))
    output = (output - np.mean(output, axis=0)) / np.std(output, axis=0)

    return np.hstack((np.ones((data.shape[0], 1)), output))


def pca(train_data, test_data, level=0.99):
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
            value = np.nanmedian(train_data[np.where(index == 0), i])
            train_data[np.where(index == 1), i] = value
            text_data[np.where(text_data[:, i] == -999), i] = value
            i = i + 1
        index = np.zeros((len(train_data), 1))
    return train_data, text_data


def split_data_by_categories(x, y, test_data, test_id):
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


def preprocess(train_data, train_target, test_data, ids, split=True, std_=True, type='z-norm', poly=True,
               degree=1, fix_emp=True, embedding=True, t=0.99):
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
        else:
            for i in range(len(xs)):
                xs[i], ts[i] = min_max_std(xs[i], ts[i])

    if poly:
        xs = [build_poly(xs[i], degree) for i in range(len(xs))]
        ts = [build_poly(ts[i], degree) for i in range(len(ts))]

    if embedding:
        for i in range(len(xs)):
            xs[i], ts[i] = pca(xs[i], ts[i], t)

    return xs, ys, ts, ids


def cross_validation(x, y, n_fold=5, f='gradient_descent', lambda_=0.0, epochs=300, gamma=1e-1, p=False, batch_size=50):
    a = 0
    step = int(0.9+len(x) / n_fold)
    result = 0
    count=0
    while a < len(x):
        end = a + step
        if end > len(x):
            end = len(x)
        data = np.concatenate((x[:a], x[end:]), axis=0)
        targets = np.concatenate((y[:a], y[end:]), axis=0)
        test_data, test_targets = x[a:end], y[a:end]
        w = None
        if f == 'gradient_descent':
            loss, w = gradient_descent(targets, data, np.zeros((data.shape[1], targets.shape[1])), iteration=epochs,
                                       gamma=gamma, print_output=p)

        elif f == 'stochastic_gradient_descent':
            loss, w = stochastic_gradient_descent(targets, data, np.zeros((data.shape[1], targets.shape[1])),
                                                  iteration=epochs, gamma=gamma, batch_size=batch_size, print_output=p)

        elif f == 'least_squares':
            loss, w = least_squares(targets, data)

        elif f == 'ridge_regression':
            loss, w = ridge_regression(targets, data, lambda_)

        elif f == 'logistic_regression':
            loss, w = logistic_regression(targets, data,
                                          np.zeros((data.shape[1], targets.shape[1])), iteration=epochs, gamma=gamma,
                                          print_output=p)
        elif f == 'reg_logistic_regression':
            loss, w = reg_logistic_regression(targets, data,
                                              np.zeros((data.shape[1], targets.shape[1])), iteration=epochs,
                                              gamma=gamma, print_output=p, lambda_=lambda_)
        if f == 'dnn':
            net = Sequential(Linear(data.shape[1], 128), ReLU(), Linear(128, 128), ReLU(),
                             Linear(128, targets.shape[1]))

            train_model(data, targets, net, print_res=p, nb_epochs=epochs, lambda_l2=lambda_, learning_rate=gamma,
                        test_data=test_data, test_target=test_targets)
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
