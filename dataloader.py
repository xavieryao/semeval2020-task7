import csv
import re
import random
import torch
import pickle
from torch.utils.data import Dataset


class HeadlineDataset(Dataset):
    def __init__(self, dataset='training'):
        assert dataset in ('training', 'dev')
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
        return torch.stack(word_vectors)


def DataLoader(dataset, batch_size, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for i in range((len(dataset) + batch_size - 1) // batch_size):
        batch_indices = indices[i*batch_size: (i+1)*batch_size]
        batch = [dataset[x] for x in batch_indices]
        batch = torch.nn.utils.rnn.pack_sequence(batch, enforce_sorted=False)
        yield batch


if __name__ == '__main__':
    ds = HeadlineDataset('dev')
    dl = DataLoader(ds, 10)
    for s in dl:
        print(s)
