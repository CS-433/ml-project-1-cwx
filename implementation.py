import numpy as np
from numpy import linalg


def compute_loss(y, x, w):
    """Calculate the loss using mse."""
    error = y - x.dot(w)

    return np.sum(np.power(error, 2)) / (2 * len(y))


def compute_gradient(y, x, w):
    """Compute the gradient."""

    error = y - x.dot(w)

    return -1.0 * x.transpose().dot(error) / len(y)


def gradient_descent(y, x, wi, iteration, gamma, print_output=False):
    """Gradient descent algorithm."""
    ws = [wi]
    losses = [compute_loss(y, x, wi)]
    w = wi
    for n_iter in range(iteration):
        grad = compute_gradient(y, x, w)
        w = w - gamma * grad
        loss = compute_loss(y, x, w)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]


def compute_stoch_gradient(y, x, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - x.dot(w)
    return np.transpose([-1.0 * e * x])


def stochastic_gradient_descent(y, x, wi, iteration, gamma, batch_size=1, print_output=False):
    """Stochastic gradient descent algorithm."""
    ws = [wi]
    losses = [compute_loss(y, x, wi)]
    w = wi
    for n_iter in range(iteration):
        i = np.random.choice(len(y))
        grad = compute_stoch_gradient(y[i], x[i], w)
        for m_iter in range(batch_size - 1):
            i = np.random.choice(len(y))
            grad = grad + compute_stoch_gradient(y[i], x[i], w)

        grad = grad / batch_size
        w = w - gamma * grad
        loss = compute_loss(y, x, w)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]


def compute_stoch_gradient2(y, x, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - x.dot(w)
    return np.transpose(-1.0 * e.transpose().dot(x)) / len(y)


def stochastic_gradient_descent2(y, x, wi, iteration, gamma, batch_size=50, print_output=False):
    """Stochastic gradient descent algorithm."""
    ws = [wi]
    losses = [compute_loss(y, x, wi)]
    w = wi
    for n_iter in range(iteration):
        a = 0
        b = batch_size
        while a < len(x):
            if b >= len(y):
                b = len(y)
            grad = compute_stoch_gradient2(y[a:b], x[a:b], w)
            w = w - gamma * grad
            a = b
            b = b + batch_size

        loss = compute_loss(y, x, w)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]


def adam1(y, x, wi, iteration, gamma, print_output=False, p1=0.9, p2=0.999, eps=1e-8):
    """Gradient descent algorithm using adaptive momentum estimation."""
    ws = [wi]
    losses = [compute_loss(y, x, wi)]
    w = wi
    v = 0
    m = 0
    for n_iter in range(iteration):
        grad = compute_gradient(y, x, w)
        m = p1 * m + (1 - p1) * grad
        v = p2 * v + (1 - p2) * (grad ** 2)
        w = w - gamma * m / (np.sqrt(v) + eps)
        loss = compute_loss(y, x, w)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]


def adam2(y, x, wi, iteration, gamma, batch_size=1, print_output=False, p1=0.9, p2=0.999, eps=1e-8):
    """Stochastic gradient descent algorithm using adaptive momentum estimation."""
    ws = [wi]
    losses = [compute_loss(y, x, wi)]
    w = wi
    v = 0
    m = 0
    g = gamma
    for n_iter in range(iteration):
        a = 0
        b = batch_size
        g = gamma / (2 * (n_iter + 1))
        while a < len(x):
            if b >= len(y):
                b = len(y)
            grad = compute_stoch_gradient2(y[a:b], x[a:b], w)
            m = p1 * m + (1 - p1) * grad
            v = p2 * v + (1 - p2) * (grad ** 2)
            w = w - g * m / (np.sqrt(v) + eps)
            a = b
            b = b + batch_size

        loss = compute_loss(y, x, w)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]


def least_squares(y, x):
    """"Least squares regression using normal equations"""
    g = x.transpose().dot(x)
    w = linalg.inv(g).dot(x.transpose()).dot(y)
    return compute_loss(y, x, w), w


def ridge_regression(y, x, lambda_=0):
    """Ridge regression using normal equations"""
    g = x.transpose().dot(x) + 2 * x.shape[0] * lambda_
    w = linalg.inv(g).dot(x.transpose()).dot(y)
    return compute_loss(y, x, w), w


def calculate_loss(y, x, w, lambda_=0):
    """compute the cost function"""
    res = 0.00
    for i in range(len(y)):
        res = res + np.log(1 + np.exp(x[i].transpose().dot(w))) - y[i] * x[i].transpose().dot(w)
    res = res / len(y)
    return res + lambda_ * w.transpose().dot(w) / 2


def sigmod(x):
    """compute sigmod function"""
    return np.exp(x) / (1 + np.exp(x))


def std(x):
    """standardization."""
    x = x - np.mean(x)
    x = x / np.std(x)
    return x


def logistic_gradient(y, x, w, lambda_=0):
    """compute gradient"""
    return x.transpose().dot(sigmod(x.dot(w)) - y) + lambda_ * w


def logistic_regression(y, x, wi, iteration, gamma, print_output=False):
    """Gradient ascent for optimal parameters"""
    ws = [wi]
    losses = [calculate_loss(y, x, wi)]

    w = wi
    for i in range(iteration):
        grad = logistic_gradient(y, x, w)
        w = w - gamma * grad
        loss = calculate_loss(y, x, w)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=i, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]


def reg_logistic_regression(y, x, wi, iteration, gamma, lambda_, print_output=False):
    """Gradient ascent for optimal parameters"""
    ws = [wi]
    losses = [calculate_loss(y, x, wi)]
    w = wi
    for i in range(iteration):
        grad = logistic_gradient(y, x, w, lambda_)
        w = w - gamma * grad

        loss = calculate_loss(y, x, w, lambda_)
        ws.append(w)
        losses.append(loss)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=i, ti=iteration - 1, l=loss))

    return losses[-1], ws[-1]
