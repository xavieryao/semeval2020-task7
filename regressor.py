from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import HuberRegressor
import numpy as np
import pickle
from dataloader import HeadlineDataset
from csv import writer
import os, subprocess
import math
from collections import Counter


def get_weights(ys, countings):
    total = sum(countings.values())
    weights = []
    for y in ys:
        bin_num = int(y * 5)
        weights.append(total / countings[bin_num])
    print(ys[:10])
    print(weights[:10])
    return weights


def load_data(ds):
    with open(f'data/{ds}_features.pkl', 'rb') as f:
        train_features = pickle.load(f)

    with open(f'data/{ds}-lstm.csv') as f:
        lines = f.readlines()[1:]
        for i, line in enumerate(lines):
            lstm_prediction = float(line.split(',')[1])
            train_features[i][0]['lstm-output'] = lstm_prediction
        
    Xs = []
    Ys = []
    for features, y in train_features:
        Ys.append(y)
        x = []
        for k in ["orig_score", "edit_score", "bert_sim", "glove_sim", "score_diff", "lstm-output"]:
            x.append(features[k])
        x = np.array(x)
        Xs.append(x)


    return Xs, Ys

# grouping bins
countings = Counter()
for i in range(30):
    countings[i] += 1
dev_dataset = HeadlineDataset('dev')
train_dataset = HeadlineDataset('training')
for sample in dev_dataset:
    bin_num = int(sample['label'] * 5)
    countings[bin_num] += 1
for sample in train_dataset:
    bin_num = int(sample['label'] * 5)
    countings[bin_num] += 1


print('load data')
Xs, Ys = load_data('train')
train_weights = get_weights(Ys, countings)
dev_Xs, dev_Ys = load_data('dev')
dev_weights = get_weights(dev_Ys, countings)


model = GradientBoostingRegressor(
    learning_rate=0.05, n_estimators=50,
    subsample=0.5,
    min_samples_split=2,
    max_depth=3
)

print('train')
model.fit(Xs, Ys, train_weights)
print('trained')

print(model.feature_importances_)

pred_Ys = model.predict(dev_Xs)
dev_rmse = math.sqrt(mean_squared_error(dev_Ys, pred_Ys))
print(dev_rmse)

test_Xs, _ = load_data('test')
pred_Ys = model.predict(test_Xs)
test_dataset = HeadlineDataset('test')
with open('data/task-1-output.csv', 'w') as f:
    output_writer = writer(f)
    output_writer.writerow(('id', 'pred'))
    for row, pred in zip(test_dataset, pred_Ys):
        output_writer.writerow((row['id'], pred.item()))
os.chdir('data')
subprocess.run(['zip', 'task-1-output.zip', 'task-1-output.csv'])
os.chdir('..')