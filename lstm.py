from torch import nn
from torch.nn import functional as F


class LSTMBaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=16)
        self.linear1 = nn.Linear(16, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        seq_len, batch, input_size = x.shape
        _, (x, _) = self.lstm(x)  # take the last hidden state
        x = x.view(batch, 16)
        x = F.dropout(x, 0.5)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)

        # predict the score
        x = self.linear2(x)
        x = 4 * F.sigmoid(x)  # normalize it to [0, 4]
        return x