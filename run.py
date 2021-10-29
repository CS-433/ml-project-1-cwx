from proj1_helpers import *
from util import *
"""Load data"""
train_path = "data/train.csv/train.csv"
test_path = "data/test.csv/test.csv"
train_target, train_data, _ = load_csv_data(train_path)
_, test_data, ids = load_csv_data(test_path)

"""Data processing"""
train_target = train_target.reshape((len(train_target), 1))
train_data, test_data = fix_empty(train_data, test_data)
train_data, test_data = standardize(train_data, test_data)
train_target = convert_to_one_hot(train_target)

"""Create dnn model and train the model"""
model = Sequential(Linear(train_data.shape[1], 128), ReLU(), Linear(128, 128), ReLU(),
                   Linear(128, train_target.shape[1]))
train_model(train_data, train_target, model, crit='mse', print_res=False, nb_epochs=480,
            mini_batch_size=100, lambda_l2=8e-6, learning_rate=1e-1, cosine=True)

"""Predict test_data and create submission file"""
output = model.forward(test_data)
predicted = np.ones((len(output), 1), dtype=int)
predicted[np.where(output[:, 0] < output[:, 1])] = -1
create_csv_submission(ids, predicted, 'submission')