import csv
import re
import random
import torch
import pickle
from torch.utils.data import Dataset


class HeadlineDataset(Dataset):
    def __init__(self, dataset='training'):
        assert dataset in ('training', 'dev', 'test')
        self.dataset = dataset
        dataset_path = f'data/{dataset}.csv'
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        with open('data/glove.pkl', 'rb') as f:
            self.glove_embeddings = pickle.load(f)

    @staticmethod
    def get_edited_text(row):
        orig = row['original']
        edited = re.sub(r"<.+/>", row['edit'], orig)
        return edited

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        headline = self.get_edited_text(self.rows[idx])
        word_vectors = []
        for word in headline.split():
            word_vectors.append(self.glove_embeddings.get(word, self.glove_embeddings['<UNK>']))
        res = {
            'id': self.rows[idx]['id'],
            'edited_headline_embedding': torch.stack(word_vectors)
        }
        if 'meanGrade' in self.rows[idx]:
            res['label'] = float(self.rows[idx]['meanGrade'])
            res['5bin-label'] = int(self.rows[idx]['5bin-label']),
            res['10bin-label'] = int(self.rows[idx]['10bin-label'])
        return res


def DataLoader(dataset, batch_size=None, shuffle=True, predict=False, pad_or_pack='pad', task='regression'):
    def prepare_batch(batch):
        if pad_or_pack == 'pack':
            xs = torch.nn.utils.rnn.pack_sequence([x['edited_headline_embedding'] for x in batch], enforce_sorted=False)
        else:  # pad
            xs = torch.nn.utils.rnn.pad_sequence([x['edited_headline_embedding'] for x in batch])
        if not predict:
            if task == 'regression':
                ys = torch.Tensor([x['label'] for x in batch])
            elif task == '5-classification':
                ys = torch.LongTensor([x['5bin-label'] for x in batch])
            elif task == '10-classification':
                ys = torch.LongTensor([x['10bin-label'] for x in batch])
            else:
                raise ValueError("Invalid task")
        else:
            ys = None
        return xs, ys

    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    if batch_size:
        for i in range((len(dataset) + batch_size - 1) // batch_size):
            batch_indices = indices[i*batch_size: (i+1)*batch_size]
            batch = [dataset[x] for x in batch_indices]
            yield prepare_batch(batch)
    elif not shuffle:
        batch = dataset
    else:
        batch = [dataset[x] for x in indices]
    yield prepare_batch(batch)


if __name__ == '__main__':
    ds = HeadlineDataset('dev')
    dl = DataLoader(ds, 10)
    for s in dl:
        print(s)
