from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.tensorboard import SummaryWriter



class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class LSTMBaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=16)
        self.linear1 = nn.Linear(16, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        _, (x, _) = self.lstm(x)  # take the last hidden state
        x = x.squeeze()
        x = F.dropout(x, 0.5)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)

        # predict the score
        x = self.linear2(x)
        x = 4 * torch.sigmoid(x)  # normalize it to [0, 4]
        return x.squeeze()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def train():
    train_writer = SummaryWriter('runs/simple_lstm_exp_1_train')
    dev_writer = SummaryWriter('runs/simple_lstm_exp_1_dev')

    from dataloader import DataLoader, HeadlineDataset
    print('Loading data')
    train_dataset = HeadlineDataset('training')
    dev_dataset = HeadlineDataset('dev')

    net = LSTMBaselineModel()
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        trainloader = DataLoader(train_dataset, 8)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 20))
                train_writer.add_scalar('epoch', epoch + 1, epoch * len(train_dataset) + i)
                train_writer.add_scalar('loss', running_loss / 20, epoch * len(train_dataset) + i)
                running_loss = 0.0
            if i % 50 == 49:  # validate
                dev_loader = DataLoader(dev_dataset)
                dev_xs, dev_ys = next(dev_loader)
                val_outputs = net(dev_xs)
                val_loss = criterion(val_outputs, dev_ys)
                print('[%d, %5d] val loss: %.6f' % (epoch + 1, i + 1, val_loss))
                dev_writer.add_scalar('loss', val_loss, epoch * len(train_dataset) + i)
        # save checkpoint
        net.save("checkpoints/{}.pyt".format(epoch+1))
    train_writer.close()
    dev_writer.close()
    print('Finished Training')


if __name__ == '__main__':
    train()