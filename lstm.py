from torch import nn
from torch.nn import functional as F
import torch


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
        return x


if __name__ == '__main__':
    from dataloader import DataLoader, HeadlineDataset
    ds = HeadlineDataset()
    dl = DataLoader(ds, 10)
    xs, ys = next(dl)
    net = LSTMBaselineModel()
    print(net(xs))
    print(ys)