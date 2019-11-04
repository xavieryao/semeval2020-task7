from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import DataLoader, HeadlineDataset
import subprocess
import os
from configs import *


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class LSTMWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, training):
        seq_len = x.shape[0]
        h, c = self.cell(x[0])
        h = F.dropout(h, self.dropout, training=training)
        for i in range(1, seq_len):
            h, c = self.cell(x[i], (h, c))
            h = F.dropout(h, self.dropout, training=training)
        return h


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
        self.lstm = LSTMWithDropout(input_size=300, hidden_size=config['hidden'], dropout=config['dropout'])
        self.lstm_norm = nn.BatchNorm1d(config['hidden'])
        self.linear1 = nn.Linear(config['hidden'], config['linear'])
        self.linear_norm = nn.BatchNorm1d(config['linear'])
        self.linear2 = nn.Linear(config['linear'], 1)

    def forward(self, x, training=True):
        x = self.lstm(x, training=training)  # take the last hidden state
        x = x.squeeze()
        if self.config['norm']:
            x = self.lstm_norm(x)
        x = self.linear1(x)
        if self.config['norm']:
            x = self.linear_norm(x)
        x = F.relu(x)
        x = F.dropout(x, self.config['dropout'], training=training)

        # predict the score
        x = self.linear2(x)
        x = 4 * torch.sigmoid(x)  # normalize it to [0, 4]
        return x.squeeze()


def train():
    train_writer = SummaryWriter('runs/simple_lstm_exp_1_train')
    dev_writer = SummaryWriter('runs/simple_lstm_exp_1_dev')

    print('Loading data')
    train_dataset = HeadlineDataset('training')
    dev_dataset = HeadlineDataset('dev')

    net = LSTMBaselineModel(original_lstm)
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        trainloader = DataLoader(train_dataset, 8)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if len(labels) <= 1:
                continue

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
            if i % 50 == 49:  # validate
                dev_loader = DataLoader(dev_dataset)
                dev_xs, dev_ys = next(dev_loader)
                val_outputs = net(dev_xs, training=False)
                val_loss = criterion(val_outputs, dev_ys)
                print('[%d, %5d] val loss: %.6f' % (epoch + 1, i + 1, val_loss))
                dev_writer.add_scalar('loss', val_loss, epoch * len(train_dataset) + + min(i * 8, len(train_dataset)))
        # save checkpoint
        net.save("checkpoints/{}.pyt".format(epoch+1))
    train_writer.close()
    dev_writer.close()
    print('Finished Training')


def predict():
    from csv import writer
    test_dataset = HeadlineDataset('test')
    test_loader = DataLoader(test_dataset, shuffle=False, predict=True)
    net = LSTMBaselineModel(larger_lstm)
    net.load('checkpoints/7.pyt')
    xs, _ = next(test_loader)
    y_pred = net(xs, training=False)
    with open('data/task-1-output.csv', 'w') as f:
        output_writer = writer(f)
        output_writer.writerow(('id', 'pred'))
        for row, pred in zip(test_dataset, y_pred):
            output_writer.writerow((row['id'], pred.item()))
    os.chdir('data')
    subprocess.run(['zip', 'task-1-output.zip', 'task-1-output.csv'])
    os.chdir('..')


if __name__ == '__main__':
    import sys
    if 'pred' in sys.argv:
        predict()
    else:
        train()