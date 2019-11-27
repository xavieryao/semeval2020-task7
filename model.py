from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import DataLoader, HeadlineDataset
import subprocess
import os
from configs import *
import json
from glob import glob
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load bin stats
bin_stats = {}
for stat_file in glob('data/*bin-stat.json'):
    with open(stat_file) as f:
        bin_stats[stat_file.split('/')[1].split('bin')[0]] = json.load(f)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class LSTMWithStepwiseDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        if self.dropout == 0 or config.get('skip_lstm_dropout', False):
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
            self.cell = None
        else:
            self.lstm = None
            self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, training):
        seq_len = x.shape[0]
        if self.cell:
            h, c = self.cell(x[0])
            h = F.dropout(h, self.dropout, training=training)
            for i in range(1, seq_len):
                h, c = self.cell(x[i], (h, c))
                h = F.dropout(h, self.dropout, training=training)
            return h
        else:
            _, (h, _) = self.lstm(x)
            return h

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}, dropout={self.dropout}"


class SavableModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class LSTMBaselineModel(SavableModel):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.lstm = LSTMWithStepwiseDropout(input_size=300, hidden_size=config['hidden'], dropout=config['dropout'])
        self.linear1 = nn.Linear(config['hidden'], config['linear'])
        self.linear_norm = nn.BatchNorm1d(config['linear'])
        if config['task'] == 'regression':
            self.linear2 = nn.Linear(config['linear'], 1)
        elif config['task'].endswith('classification'):
            self.linear2 = nn.Linear(config['linear'], int(config['task'].split('-')[0]))

    def forward(self, x, training=True):
        x = self.lstm(x, training=training)  # take the last hidden state
        x = x.squeeze()
        x = self.linear1(x)
        if self.config['norm']:
            x = self.linear_norm(x)
        x = F.relu(x)
        x = F.dropout(x, self.config['dropout'], training=training)

        # predict the score
        x = self.linear2(x)
        if config['task'] == 'regression':
            x = 3 * torch.sigmoid(x)  # normalize it to [0, 3]
            x = x.squeeze()
        return x


def calculate_val_rmse(val_set, outputs, task):
    # calculate validation RMSE for bin-classification
    bin_stat = bin_stats[task.split('-')[0]]
    assignments = [x['avg'] for x in bin_stat]
    pred_bins = torch.argmax(outputs, dim=1)
    loss = 0.
    for bin_idx, row in zip(pred_bins, val_set):
        y_pred = assignments[bin_idx]
        y_true = row['label']
        loss += (y_pred - y_true) ** 2
    loss = (loss / len(val_set)) ** 0.5
    return loss


def train(config):
    train_writer = SummaryWriter('runs/simple_lstm_exp_1_train')
    dev_writer = SummaryWriter('runs/simple_lstm_exp_1_dev')

    print('Loading data')
    train_dataset = HeadlineDataset('training')
    dev_dataset = HeadlineDataset('dev')

    net = LSTMBaselineModel(config)
    net = net.to(device)
    print(net)
    if config['task'] == 'regression':
        criterion = RMSELoss()
    else:  # config['task'] == 'B-classification'
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    best_val_rmse = float('+inf')
    for epoch in range(300):  # loop over the dataset multiple times
        running_loss = 0.0
        trainloader = DataLoader(train_dataset, 64, task=config['task'])
        for i, data in enumerate(trainloader):
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if len(labels.shape) == 0 or len(labels) <= 1:
                continue
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 20))
                train_writer.add_scalar('epoch', epoch + 1, epoch * len(train_dataset) + min(i * 8, len(train_dataset)))
                train_writer.add_scalar('loss', running_loss / 20, epoch * len(train_dataset) + min(i * 8, len(train_dataset)))
                running_loss = 0.0
            if i % 20 == 19:  # validate
                net.eval()
                dev_loader = DataLoader(dev_dataset, task=config['task'], shuffle=False)
                dev_xs, dev_ys = next(dev_loader)
                dev_xs = dev_xs.to(device)
                dev_ys = dev_ys.to(device)
                val_outputs = net(dev_xs, training=False)
                val_loss = criterion(val_outputs, dev_ys)
                print('[%d, %5d] val loss: %.6f' % (epoch + 1, i + 1, val_loss))
                dev_writer.add_scalar('loss', val_loss, epoch * len(train_dataset) + + min(i * 8, len(train_dataset)))

                if config['task'] == 'regression':
                    val_rmse = val_loss
                else:
                    val_rmse = calculate_val_rmse(dev_dataset, val_outputs, config['task'])

                dev_writer.add_scalar('val rmse', val_rmse,
                                      epoch * len(train_dataset) + + min(i * 8, len(train_dataset)))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    net.save("checkpoints/best.pyt".format(epoch+1))
                    dev_writer.add_scalar('best val rmse', val_rmse, epoch * len(train_dataset) + + min(i * 8, len(train_dataset)))
        # save checkpoint
        net.save("checkpoints/{}.pyt".format(epoch+1))
    train_writer.close()
    dev_writer.close()
    print('Finished Training')


def predict(config):
    from csv import writer
    test_dataset = HeadlineDataset('training')
    test_loader = DataLoader(test_dataset, shuffle=False, predict=True, task=config['task'])
    net = LSTMBaselineModel(config)
    net.load('checkpoints/best.pyt')
    net.eval()
    xs, _ = next(test_loader)
    y_pred = net(xs, training=False)
    with open('data/train-lstm.csv', 'w') as f:
        output_writer = writer(f)
        output_writer.writerow(('id', 'pred'))
        for row, pred in zip(test_dataset, y_pred):
            output_writer.writerow((row['id'], pred.item()))
    os.chdir('data')
    subprocess.run(['zip', 'task-1-output.zip', 'task-1-output.csv'])
    os.chdir('..')


if __name__ == '__main__':
    config = larger_lstm
    import shutil
    try:
        shutil.rmtree('runs')
    except:
        pass

    import sys
    if 'pred' in sys.argv:
        predict(config)
    else:
        train(config)
