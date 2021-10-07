import numpy as np

from proj1_helpers import *


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        None


class Linear(Module):
    def __init__(self, i_dim, o_dim):
        super(Linear, self).__init__()
        e = 1 / np.sqrt(i_dim)
        self.w = np.random.uniform(-e, e, (o_dim, i_dim))
        self.b = np.random.uniform(-e, e, o_dim)
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        self.x = 0
        self.m1 = np.zeros(self.w.shape)
        self.v1 = np.zeros(self.w.shape)
        self.m2 = np.zeros(self.b.shape)
        self.v2 = np.zeros(self.b.shape)

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
        return [[self.w, self.dw, self.m1, self.v1], [self.b, self.db, self.m2, self.v2]]

    def zero_grad(self):
        self.dw = np.zeros(self.dw.shape)
        self.db = np.zeros(self.db.shape)


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


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.x = None

    def forward(self, x, test=False):
        self.x = x
        return np.tanh(x)

    def backward(self, gradwrtoutput):
        return 4 * (np.power(np.exp(self.x) + np.exp(-1 * self.x), -2)) * gradwrtoutput


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


class SGD(Module):

    def __init__(self, modules, learning_rate=0.001):
        self.p = modules.param()
        self.lr = learning_rate

    def step(self):
        for (p, dp) in self.p:
            p = p - self.lr * dp


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

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


def compute_error(target, prediction):
    count = 0
    for i in range(target.shape[0]):
        if target[i, prediction[i]] != 1:
            count = count + 1
    return count


def train_model(train_input, train_target, model, lambda_l2=0, learning_rate=1e-1, nb_epochs=1000,
                mini_batch_size=1000, print_res=False):
    criterion = MSE()
    optimizer = SGD(model, learning_rate)
    p1 = 0.9
    p2 = 0.999
    eps = 1e-8
    index = np.arange(0, train_input.shape[0], 1, dtype=int)
    for e in range(0, nb_epochs + 1):
        loss = 0
        a = 0
        b = mini_batch_size
        while a < mini_batch_size:
            if b >= len(train_input):
                b = len(train_input)
            output = model.forward(train_input[np.where(index % mini_batch_size == a)])
            loss += criterion.forward(output, train_target[np.where(index % mini_batch_size == a)])
            model.zero_grad()
            model.backward(criterion.backward())
            for (p, dp, m, v) in model.param():
                p *= (1 - lambda_l2)
                p -= learning_rate * dp
            a = a + 1
            b = b + mini_batch_size

        if e % 10 == 0 and print_res:
            print(e)
            output = model.forward(x1)
            predicted = np.ones((len(output), 1), dtype=int)
            predicted[np.where(output[:, 0] > output[:, 1])] = 0
            print("train", compute_error(y1, predicted) / x1.shape[0])
            output = model.forward(x2)
            predicted = np.ones((len(output), 1), dtype=int)
            predicted[np.where(output[:, 0] > output[:, 1])] = 0
            print("test", compute_error(y2, predicted) / x2.shape[0])


train_path = "data/train.csv/train.csv"
test_path = "data/test.csv/test.csv"
train_target, train_data, _ = load_csv_data(train_path)
train_target = train_target.reshape((len(train_target), 1))
_, text_data, ids = load_csv_data(test_path)

train_data, text_data = fix_empty(train_data, text_data)
train_data, text_data = standardize(train_data, text_data)
train_data, text_data = pca(train_data, text_data)
train_target = convert_to_one_hot(train_target)

y1, x1, y2, x2 = train_target[:200000], train_data[:200000], train_target[200000:], train_data[200000:]

model = Sequential(Linear(x1.shape[1], 64), ReLU(), Linear(64, 16), ReLU(), Linear(16, 16), ReLU(),
                   Linear(16, y1.shape[1]))

train_model(train_data, train_target, model, print_res=True)

output = model.forward(x1)
predicted = np.ones((len(output), 1), dtype=int)
predicted[np.where(output[:, 0] > output[:, 1])] = 0
print("train_error", compute_error(y1, predicted) / y1.shape[0])

output = model.forward(x2)
predicted = np.ones((len(output), 1), dtype=int)
predicted[np.where(output[:, 0] > output[:, 1])] = 0
print("test_error", compute_error(y2, predicted) / y2.shape[0])
