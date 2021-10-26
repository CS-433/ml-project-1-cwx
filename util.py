import numpy as np

from proj1_helpers import *

"""The class Module represents the basic structure of each module"""
class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        None


"""The module Linear defines a fully connected layer, applying a linear transformation to the input data"""
class Linear(Module):
    def __init__(self, i_dim, o_dim):
        super(Linear, self).__init__()
        e = 1 / np.sqrt(i_dim)
        self.w = np.random.uniform(-e, e, (o_dim, i_dim))
        self.b = np.random.uniform(-e, e, o_dim)
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.x = 0

    def forward(self, x, test=False):
        self.x = x
        x = x.dot(self.w.transpose())
        x = x + self.b
        return x

    def backward(self, gradwrtoutput):
        gradwrtoutput = gradwrtoutput.transpose()
        self.dw += gradwrtoutput.dot(self.x)
        self.db += gradwrtoutput.sum(1)
        return (self.w.transpose().dot(gradwrtoutput)).transpose()

    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]

    def zero_grad(self):
        self.dw = np.zeros(self.dw.shape)
        self.db = np.zeros(self.db.shape)


"""The module ReLu applies the rectified linear unit activation function to convert input value into positive numbers"""
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.x = None

    def forward(self, x, test=False):
        self.x = x
        x[x < 0] = 0
        return x

    def backward(self, gradwrtoutput):
        x = self.x
        x[x > 0] = 1
        x[x < 0] = 0
        return np.multiply(gradwrtoutput, x)


"""The module Tanh applies the hyperbolic tangent activation function tanh to map the input values in range(-1,1)"""
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.x = None

    def forward(self, x, test=False):
        self.x = x
        return np.tanh(x)

    def backward(self, gradwrtoutput):
        return 4 * (np.power(np.exp(self.x) + np.exp(-1 * self.x), -2)) * gradwrtoutput


"""The module Dropout set a random part of input data to zero to reduce overfitting"""
class Dropout(Module):
    def __init__(self, prob=0.0):
        super(Dropout, self).__init__()
        self.mask = None
        self.prob = prob

    def forward(self, x, test=False):
        if test:
            return x
        self.mask = np.random.binomial(1, self.prob, x.shape)
        out = np.multiply(x, self.mask) / (1 - self.prob)
        return out

    def backward(self, gradwrtoutput):
        return np.multiply(gradwrtoutput, self.mask) / (1 - self.prob)


"""The module MSE deals with the computation of the mean squared error"""
class MSE(Module):
    def __init__(self):
        self.error = None
        self.n = None

    def forward(self, pred, label):
        self.error = pred - label
        self.n = pred.shape[0]
        return np.mean(np.power(self.error, 2))

    def backward(self):
        return self.error / self.n


"""The module SGD executes the stochastic gradient descent to optimize the parameters of the model"""
class SGD(Module):

    def __init__(self, modules, learning_rate=0.001):
        self.p = modules.param()
        self.lr = learning_rate

    def step(self):
        for (p, dp) in self.p:
            p = p - self.lr * dp


"""The module Sequential builts the sequential structures by combining different modules"""
class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = []
        for mod in modules:
            self.modules.append(mod)

    def add_module(self, module):
        self.modules.append(module)

    def forward(self, input, test=False):
        out = input
        for module in self.modules:
            out = module.forward(out, test=False)
        return out

    def backward(self, grdwrtoutput):
        out = grdwrtoutput
        for module in reversed(self.modules):
            out = module.backward(out)

    def param(self):
        parameters = []
        for module in self.modules:
            for p in module.param():
                parameters.append(p)
        return parameters

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


"""Compute the number of errors in the prediction"""
def compute_error(target, prediction):
    count = 0
    for i in range(target.shape[0]):
        if target[i, prediction[i]] != 1:
            count = count + 1
    return count


"""Train the model with cosine annealing algorithm"""
def train_model(input, target, model, lambda_l2=0.0, learning_rate=1e-1, nb_epochs=300, T=100,
                mini_batch_size=1000, print_res=False, test_data=None, test_target=None):
    criterion = MSE()
    for e in range(0, nb_epochs + 1):
        lr = 0.5 * (1 + np.cos(np.pi * e / T)) * learning_rate
        loss = 0
        a = 0
        b = mini_batch_size
        while a < len(input):
            if b >= len(input):
                b = len(input)

            output = model.forward(input[a:b])
            loss += criterion.forward(output, target[a:b])
            model.zero_grad()
            model.backward(criterion.backward())
            for (p, dp) in model.param():
                p *= (1 - lambda_l2)
                p -= lr * dp
            a = b
            b = b + mini_batch_size
        if print_res:
            print("epoch", e, "loss", loss, "lr", lr)
            if e % 10 == 0:
                output = model.forward(input)
                predicted = np.ones((len(output), 1), dtype=int)
                predicted[np.where(output[:, 0] > output[:, 1])] = 0
                print("train_error", compute_error(target, predicted) / input.shape[0])
                output = model.forward(test_data)
                predicted = np.ones((len(output), 1), dtype=int)
                predicted[np.where(output[:, 0] > output[:, 1])] = 0
                print("test_error", compute_error(test_target, predicted) / test_data.shape[0])
