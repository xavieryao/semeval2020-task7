import csv
import re
import random
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class HeadlineDataset(Dataset):
    def __init__(self, dataset):
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
            word_vectors.append(self.glove_embeddings.get(word, '<OOV>'))
        return torch.stack(word_vectors)
        """
        return {
            'edited_headline': self.get_edited_text(self.rows[idx]),
            'edited_headline_embedding': word_vectors
        }
        """