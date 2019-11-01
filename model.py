from torch import nn
from torch.nn import functional as F
import torch


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


def train():
    from dataloader import DataLoader, HeadlineDataset
    train_dataset = HeadlineDataset('training')

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
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    train()