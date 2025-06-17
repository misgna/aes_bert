import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, essays, scores, features):
        self.encodings = tokenizer(essays, padding=True, truncation=True, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float)
        self.features = torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.scores[idx]
        item['features'] = self.features[idx]
        return item
