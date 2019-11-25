import csv
import re
import random
import torch
import pickle
from torch.utils.data import Dataset
from transformers import BertTokenizer


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

    @staticmethod
    def get_orig_text(row):
        orig = row['original']
        edited = re.sub(r"<(.+)/>", r"\g<1>", orig)
        return edited

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        headline = self.get_edited_text(self.rows[idx])
        word_vectors = []
        for word in headline.split():
            word_vectors.append(self.glove_embeddings.get(word, self.glove_embeddings['<UNK>']))
        res = {
            'id': self.rows[idx]['id'],
            'edited_headline_embedding': torch.stack(word_vectors),
            'orig_text': self.get_orig_text(row),
            'edited_text': headline
        }
        if 'meanGrade' in self.rows[idx]:
            res['label'] = float(self.rows[idx]['meanGrade'])
            for k, v in self.rows[idx].items():
                if k.endswith('bin-label'):
                    res[k] = int(v)
        return res


def BertDataLoader(dataset, batch_size, shuffle=True):
    indices = list(range(len(dataset)))
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    if shuffle:
        random.shuffle(indices)
    for i in range((len(dataset) + batch_size - 1) // batch_size):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        batch = [dataset[x] for x in batch_indices]
        xs = []
        segments = []
        masks = []
        ys = []

        MAX_LEN = 80 
        for sample in batch:
            sample_segments = []
            orig_sent = '[CLS] ' + sample['orig_text'] + ' [SEP]'
            orig_tokens = tokenizer.encode(orig_sent, add_special_tokens=False)
            sample_segments.extend([0] * len(orig_tokens))
            mask = [1] * len(orig_tokens)
            mask[0] = 0  # [CLS]
            mask[-1] = 0  # [SEP]

            edited_sent = ' ' + sample['edited_text'] + ' [SEP]'
            edited_tokens = tokenizer.encode(edited_sent, add_special_tokens=False)
            sample_segments.extend([1] * len(edited_tokens))
            mask.extend([1] * len(edited_tokens))
            mask[-1] = 0  # [SEP]
            assert(len(orig_tokens) + len(edited_tokens) == len(sample_segments) == len(mask))
            
            # padding
            input_ids = orig_tokens + edited_tokens
            if len(input_ids) > MAX_LEN:
                input_ids = input_ids[:MAX_LEN]
                mask = mask[:MAX_LEN]
                sample_segments = sample_segments[:MAX_LEN]
            else:
                pad = MAX_LEN - len(input_ids)
                input_ids.extend([0] * pad)
                sample_segments.extend([0] * pad)
                mask.extend([0] * pad)


            input_ids = torch.LongTensor(input_ids)
            xs.append(input_ids)
            segments.append(torch.LongTensor(sample_segments))
            masks.append(torch.LongTensor(mask))
            if 'label' in sample:
                ys.append(sample['label'])
        xs = torch.stack(xs)
        segments = torch.stack(segments)
        masks = torch.stack(masks)
        ys = torch.Tensor(ys)
        yield xs, segments, masks, ys


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
                ys = torch.LongTensor([x['5bin-label'] for x in batch]).squeeze()
            elif task == '10-classification':
                ys = torch.LongTensor([x['10bin-label'] for x in batch]).squeeze()
            elif task.endswith('classification'):
                n_bins = task.split('-')[0]
                ys = torch.LongTensor([x[f'{n_bins}bin-label'] for x in batch]).squeeze()
            else:
                raise ValueError()
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
