from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EssayDataset(Dataset):
    def __init__(self, essays, scores):
        self.encodings = tokenizer(essays, padding=True, truncation=True, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['score'] = self.scores[idx]
        return item