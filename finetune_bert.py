import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter

from dataloader import HeadlineDataset, BertDataLoader
from model import RMSELoss, SavableModel
from shutil import copy2
import os
import subprocess


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertRegressionModel(SavableModel):
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
            x = torch.max(x, dim=-2)[0]
            x = torch.tanh(x)
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.output(x)
        x = 3 * torch.sigmoid(x)  # normalize it to [0, 3]
        x = x.squeeze()
        return x


def train(model: nn.Module):
    try:
        os.mkdir('checkpoints')
    except FileExistsError:
        pass
    try:
        os.mkdir('runs')
    except FileExistsError:
        pass

    TRAIN_LOG_FREQ = 10
    DEV_LOG_FREQ = 50

    train_writer = SummaryWriter('runs/finetune_bert_train')
    dev_writer = SummaryWriter('runs/finetune_bert_dev')
    train_dataset = HeadlineDataset('training')
    dev_dataset = HeadlineDataset('dev')
    optimizer = torch.optim.Adam(model.parameters())
    criterion = RMSELoss()

    steps = 0
    best_rmse = float('+inf')
    for epoch in range(300):
        running_loss = 0.0
        trainloader = BertDataLoader(train_dataset, 32, pair=False)
        dev_data = next(BertDataLoader(dev_dataset, len(dev_dataset), pair=False))
        for i, data in enumerate(trainloader):
            steps += len(data[0])
            model.train()
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

            if i % TRAIN_LOG_FREQ == TRAIN_LOG_FREQ - 1:
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / TRAIN_LOG_FREQ))
                train_writer.add_scalar('epoch', epoch + 1, steps)
                train_writer.add_scalar('loss', running_loss / TRAIN_LOG_FREQ, steps)
                running_loss = 0.0
            if i % DEV_LOG_FREQ == DEV_LOG_FREQ - 1: 
                model.eval()
                dev_inputs, dev_segments, dev_masks, dev_labels = [x.to(device) for x in dev_data]
                val_loss = criterion(outputs, labels)
                print('[%d, %5d]     val loss: %.6f' %
                        (epoch + 1, i + 1, val_loss))
                dev_writer.add_scalar('loss', val_loss, steps)
                # save
                model.save('checkpoints/bert_last.pt')
                if val_loss < best_rmse:
                    best_rmse = val_loss
                    copy2('checkpoints/bert_last.pt', 'checkpoints/bert_best.pt')
        # validation
        model.eval()
        dev_inputs, dev_segments, dev_masks, dev_labels = [x.to(device) for x in dev_data]
        val_loss = criterion(outputs, labels)
        print('[%d]      val loss: %.6f' %
                (epoch + 1, val_loss))
        dev_writer.add_scalar('loss', val_loss, steps)
        # save
        model.save('checkpoints/bert_last.pt')
        if val_loss < best_rmse:
            best_rmse = val_loss
            copy2('checkpoints/bert_last.pt', 'checkpoints/bert_best.pt')

def predict(model):
    from csv import writer
    test_dataset = HeadlineDataset('test')
    test_loader = BertDataLoader(test_dataset, batch_size=64)
    model.load('checkpoints/bert_best.pt')

    y_pred = []
    for data in test_loader:
        inputs, segments, masks, _ = [x.to(device) for x in data]
        ys = model(inputs, segments, masks).cpu().tolist()
        y_pred.extend(ys)
    with open('data/task-1-output.csv', 'w') as f:
        output_writer = writer(f)
        output_writer.writerow(('id', 'pred'))
        for row, pred in zip(test_dataset, y_pred):
            output_writer.writerow((row['id'], pred))
    os.chdir('data')
    subprocess.run(['zip', 'task-1-output.zip', 'task-1-output.csv'])
    os.chdir('..')


def main():
    model = BertRegressionModel()
    model = model.to(device)
    # train(model)
    predict(model)


if __name__ == '__main__':
    main()
