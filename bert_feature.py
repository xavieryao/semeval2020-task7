from dataloader import tokenizer, HeadlineDataset, convert_sent
from transformers import BertModel, BertForMaskedLM
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

model = BertModel.from_pretrained('bert-base-cased')
model.eval()

mlm_model = BertForMaskedLM.from_pretrained('bert-base-cased')
mlm_model.eval()

def find_sub_idx(full_list, sub_list):
    n = len(sub_list)
    for i in range(0, len(full_list) - n):
        if full_list[i:i+n] == sub_list:
            return i
    return -1


def get_phrase_emb(sent, phrase):
    tokens, segs, masks = convert_sent(sent)
    phrase_tokens = tokenizer.encode(phrase)
    idx = find_sub_idx(tokens.tolist(), phrase_tokens)
    assert idx != -1

    with torch.no_grad():
        outputs = model(tokens.unsqueeze(dim=0), token_type_ids=segs.unsqueeze(dim=0))
        hidden_layer = outputs[0]
    return hidden_layer[0][idx]

def get_phrase_rank(sent, phrase):
    tokens, segs, masks = convert_sent(sent)
    phrase_tokens = tokenizer.encode(phrase)
    idx = find_sub_idx(tokens.tolist(), phrase_tokens)
    tokens[idx: idx+len(phrase_tokens)] = tokenizer.mask_token_id
    assert idx != -1

    with torch.no_grad():
        outputs = mlm_model(tokens.unsqueeze(dim=0), token_type_ids=segs.unsqueeze(dim=0))
        predictions = outputs[0]
        predictions = F.softmax(predictions[0][idx], dim=-1)
    return predictions[phrase_tokens[0]]


train_ds = HeadlineDataset('test')
glove_embs = train_ds.glove_embeddings
sample = train_ds[0]
#edited_emb = get_phrase_emb(sample['edited_text'], sample['edit'])
#print(edited_emb)
def get_glove_emb(phrase):
    embs = [] 
    for w in phrase.split():
        if w in glove_embs:
            embs.append(glove_embs[w])
    if len(embs) > 0:
        return sum(embs)
    return glove_embs['<UNK>']

def extract_features(sample):
    orig_emb = get_phrase_emb(sample['orig_text'], sample['orig']).numpy()
    edit_emb = get_phrase_emb(sample['edited_text'], sample['edit']).numpy()
    orig_score = get_phrase_rank(sample['orig_text'], sample['orig'])
    edit_score = get_phrase_rank(sample['edited_text'], sample['edit'])

    orig_glove_emb = get_glove_emb(sample['orig'])
    edit_glove_emb = get_glove_emb(sample['edit'])

    score_diff = orig_score - edit_score
    emb_sim = cosine_similarity(orig_emb.reshape(1, -1), edit_emb.reshape(1, -1))
    glove_emb_sim = cosine_similarity(orig_glove_emb.reshape(1, -1), edit_glove_emb.reshape(1, -1))

    features = {
        "orig_bert_emb": orig_emb,
        "edit_bert_emb": edit_emb,
        "orig_score": orig_score,
        "edit_score": edit_score,
        "orig_glove_emb": orig_glove_emb,
        "edit_glove_emb": edit_glove_emb,
        "bert_sim": emb_sim,
        "glove_sim": glove_emb_sim,
        "score_diff": score_diff
    }

    return features, sample.get('label')


train_features = []
for i in tqdm(range(len(train_ds))):
    sample = train_ds[i]
    train_features.append(extract_features(sample))
import pickle
with open('data/test_features.pkl', 'wb') as f:
    pickle.dump(train_features, f)