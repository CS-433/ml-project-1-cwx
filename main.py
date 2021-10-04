from proj1_helpers import *
from implementation import *

print("beginning")
train_path = "data/train.csv/train.csv"
test_path = "data/test.csv/test.csv"
train_target, train_data, _ = load_csv_data(train_path)
train_target = train_target.reshape((len(train_target), 1))
_, text_data, ids = load_csv_data(test_path)

train_data, text_data = fix_empty(train_data, text_data)
train_data, text_data = standardize(train_data, text_data)
train_data = build_poly(train_data, 2)
text_data = build_poly(text_data, 2)


losses, ws = gradient_descent(train_target[:200000], train_data[:200000],
                              np.zeros((train_data.shape[1], train_target.shape[1])), 300, 1e-3, False)
print(ws[-1].shape)
print("gradient_descent:",
      "train_accuracy:", accuracy(train_target[:200000], predict_labels(ws[-1], train_data[:200000])) / 200000,
      ";  test_accuracy:",
      accuracy(train_target[200000:250000], predict_labels(ws[-1], train_data[200000:250000])) / 50000)

g_w = ws[-1]

losses, ws = stochastic_gradient_descent(train_target[:200000], train_data[:200000],
                                         np.zeros((train_data.shape[1], train_target.shape[1])), 300, 1e-2,
                                         batch_size=100,
                                         print_output=False)
print("stochastic_gradient_descent:",
      "train_accuracy:", accuracy(train_target[:200000], predict_labels(ws[-1], train_data[:200000])) / 200000,
      ";  test_accuracy:",
      accuracy(train_target[200000:250000], predict_labels(ws[-1], train_data[200000:250000])) / 50000)

sg_w = ws[-1]

w, loss = least_squares(train_target[:200000], train_data[:200000])
print("least_squares:",
      "train_accuracy:", accuracy(train_target[:200000], predict_labels(w, train_data[:200000])) / 200000,
      ";  test_accuracy:",
      accuracy(train_target[200000:250000], predict_labels(w, train_data[200000:250000])) / 50000)

l_w = w

w, loss = ridge_regression(train_target[:200000], train_data[:200000], lambda_=0.3)
print("ridge_regression:",
      "train_accuracy:", accuracy(train_target[:200000], predict_labels(w, train_data[:200000])) / 200000,
      ";  test_accuracy:",
      accuracy(train_target[200000:250000], predict_labels(w, train_data[200000:250000])) / 50000)

r_w = w

losses, ws = logistic_regression(train_target[:200000], train_data[:200000],
                                 np.zeros((train_data.shape[1], train_target.shape[1])), 10, 1e-6, False)
print("logistic_regression:",
      "train_accuracy:", accuracy(train_target[:200000], predict_labels(ws[-1], train_data[:200000])) / 200000,
      ";  test_accuracy:",
      accuracy(train_target[200000:250000], predict_labels(ws[-1], train_data[200000:250000])) / 50000)

log_w = ws[-1]

losses, ws = reg_logistic_regression(train_target[:200000], train_data[:200000],
                                     np.zeros((train_data.shape[1], train_target.shape[1])), 10, 1e-6, lambda_=0.3,
                                     print_output=False)
print("reg_logistic_regression:",
      "train_error:", accuracy(train_target[:200000], predict_labels(ws[-1], train_data[:200000])) / 200000,
      ";  test_error:",
      accuracy(train_target[200000:250000], predict_labels(ws[-1], train_data[200000:250000])) / 50000)

r_log_w = ws[-1]
