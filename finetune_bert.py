import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter

from dataloader import HeadlineDataset, BertDataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        loss = torch.norm(x - y, p=2, dim=-1)
        loss = torch.mean(loss)
        loss = torch.sqrt(loss)
        return loss


class BertRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.eval()
        self.linear = nn.Linear(self.hidden_size, 1024)
        self.output = nn.Linear(1024, 1)

    def forward(self, x, segments, masks):
        with torch.no_grad():
            x = self.bert(x, token_type_ids=segments, attention_mask=masks)[0]  # cls hidden state in the last layer
            x = torch.mean(x, dim=-2)
            x = nn.functional.tanh(x)
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.output(x)
        x = 3 * torch.sigmoid(x)  # normalize it to [0, 3]
        x = x.squeeze()
        return x


def train(model: nn.Module):
    TRAIN_LOG_FREQ = 10

    train_writer = SummaryWriter('runs/finetune_bert_train')
    dev_writer = SummaryWriter('runs/finetune_bert_dev')
    train_dataset = HeadlineDataset('training')
    optimizer = torch.optim.Adam(model.parameters())
    criterion = RMSELoss()

    for epoch in range(300):
        print('epoch', epoch)
        running_loss = 0.0
        trainloader = BertDataLoader(train_dataset, 32)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, segments, masks, labels = data
            inputs, segments, masks, labels = [x.to(device) for x in [inputs, segments, masks, labels]]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, segments, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % TRAIN_LOG_FREQ == TRAIN_LOG_FREQ - 1:  # print every 20 mini-batches
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / TRAIN_LOG_FREQ))
                train_writer.add_scalar('epoch', epoch + 1, epoch * len(train_dataset) + min(i * 8, len(train_dataset)))
                train_writer.add_scalar('loss', running_loss / TRAIN_LOG_FREQ,
                                        epoch * len(train_dataset) + min(i * 8, len(train_dataset)))
                running_loss = 0.0
            # TODO: eval, save


def main():
    model = BertRegressionModel()
    model = model.to(device)
    train(model)


if __name__ == '__main__':
    main()
