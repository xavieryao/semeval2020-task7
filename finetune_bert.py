import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
from .model import RMSELoss

from dataloader import HeadlineDataset, BertDataLoader


class BertRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.output = nn.Linear(self.hidden_size, 1)

    def forward(self, x, segments):
        x = self.bert(x, token_type_ids=segments)[0][:, 0]  # cls hidden state in the last layer
        x = 3 * torch.sigmoid(x[0])  # normalize it to [0, 3]
        x = x.squeeze()
        return x


def train(model: nn.Module):
    train_writer = SummaryWriter('runs/finetune_bert_train')
    dev_writer = SummaryWriter('runs/finetune_bert_dev')
    train_dataset = HeadlineDataset('training')
    optimizer = torch.optim.Adam(model.parameters())
    criterion = RMSELoss()

    for epoch in range(300):
        running_loss = 0.0
        trainloader = BertDataLoader(train_dataset, 64)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, segments, labels = data
            if len(labels.shape) == 0 or len(labels) <= 1:
                continue

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, segments)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 20))
                train_writer.add_scalar('epoch', epoch + 1, epoch * len(train_dataset) + min(i * 8, len(train_dataset)))
                train_writer.add_scalar('loss', running_loss / 20,
                                        epoch * len(train_dataset) + min(i * 8, len(train_dataset)))
                running_loss = 0.0
            # TODO: eval, save


def main():
    model = BertRegressionModel()
    train(model)


if __name__ == '__main__':
    main()