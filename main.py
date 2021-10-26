from proj1_helpers import *
from implementation import *

print("Start")
train_path = "data/train.csv/train.csv"
train_target, train_data, _ = load_csv_data(train_path)
train_target = train_target.reshape((len(train_target), 1))
train_data, test_data = train_data[:200000], train_data[200000:]
train_target, test_target = train_target[:200000], train_target[200000:]
train_data, test_data = standardize(train_data, test_data)


loss, w = gradient_descent(train_target, train_data,
                           np.zeros((train_data.shape[1], train_target.shape[1])), 300, 1e-3, False)
print("gradient_descent:",
      "train_accuracy:", accuracy(train_target, predict_labels(w, train_data)),
      ";  test_accuracy:",
      accuracy(test_target, predict_labels(w, test_data)))

loss, w = stochastic_gradient_descent(train_target, train_data,
                                      np.zeros((train_data.shape[1], train_target.shape[1])), 300, 1e-3,
                                      batch_size=1,
                                      print_output=False)
print("stochastic_gradient_descent:",
      "train_accuracy:", accuracy(train_target, predict_labels(w, train_data)),
      ";  test_accuracy:",
      accuracy(test_target, predict_labels(w, test_data)))

loss, w = least_squares(train_target, train_data)
print("least_squares:",
      "train_accuracy:", accuracy(train_target, predict_labels(w, train_data)),
      ";  test_accuracy:",
      accuracy(test_target, predict_labels(w, test_data)))

loss, w = ridge_regression(train_target, train_data, lambda_=0.3)
print("ridge_regression:",
      "train_accuracy:", accuracy(train_target, predict_labels(w, train_data)),
      ";  test_accuracy:",
      accuracy(test_target, predict_labels(w, test_data)))

train_target[train_target == -1] = 0
test_target[test_target == -1] = 0

loss, w = logistic_regression(train_target, train_data,
                              np.zeros((train_data.shape[1], train_target.shape[1])), 100, 1e-6, False)
print("logistic_regression:",
      "train_accuracy:", accuracy(train_target, predict_labels2(w, train_data)),
      ";  test_accuracy:",
      accuracy(test_target, predict_labels2(w, test_data)))

loss, w = reg_logistic_regression(train_target, train_data,
                                  np.zeros((train_data.shape[1], train_target.shape[1])), 100, 1e-6, lambda_=0.3,
                                  print_output=False)
print("reg_logistic_regression:",
      "train_error:", accuracy(train_target, predict_labels2(w, train_data)),
      ";  test_error:",
      accuracy(test_target, predict_labels2(w, test_data)))
