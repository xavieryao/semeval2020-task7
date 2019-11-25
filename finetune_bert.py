import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter


class BertRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.output = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.bert(x)[0][:, 0]  # cls hidden state in the last layer
        x = 3 * torch.sigmoid(x[0])  # normalize it to [0, 3]
        x = x.squeeze()
        return x