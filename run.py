from proj1_helpers import *
from implementation import *
from util import *

train_path = "data/train.csv/train.csv"
test_path = "data/test.csv/test.csv"
train_target, train_data, _ = load_csv_data(train_path)
train_target = train_target.reshape((len(train_target), 1))
_, test_data, ids = load_csv_data(test_path)
train_data, test_data = fix_empty(train_data, test_data)
train_data, test_data = standardize(train_data, test_data)
train_target = convert_to_one_hot(train_target)

model = Sequential(Linear(train_data.shape[1], 128), ReLU(), Linear(128, 128), ReLU(),
                   Linear(128, train_target.shape[1]))
train_model(train_data, train_target, model, print_res=False, nb_epochs=100, lambda_l2=1e-5, learning_rate=1e-1)
output = model.forward(test_data)
predicted = np.ones((len(output), 1), dtype=int)
predicted[np.where(output[:, 0] < output[:, 1])] = -1
create_csv_submission(ids, predicted, 'submission')
