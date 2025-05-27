import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

from sklearn.metrics import cohen_kappa_score

from transformers import BertModel, BertTokenizer, BertConfig

from bert_model import BERTRegressor
from custom_dataset import CustomDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_data(filepath):
    return pd.read_csv(filepath)

def training(model, train_loader, device, optimizer, loss_fn, EPOCHS):
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
        print(f"Epoch {epoch+1}, Training Loss: {loss.item():.4f}")

def validation(model, dev_loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    qwk = cohen_kappa_score(trues, preds, weights="quadratic")
    print(f"QWK: {qwk:.4f}")

if __name__ == "__main__":
    set_seed(42)
    train_df = load_data("datasets/xLEAF/train.csv")
    dev_df = load_data("datasets/xLEAF/dev.csv")
    #print(dev_df)
    
    EPOCHS = 2
    LR = 2e-5 # learning rate
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_df, tokenizer)
    dev_dataset = CustomDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    model = BERTRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # call training method
    training(model, train_loader, device, optimizer, loss_fn, EPOCHS)
    
    # call calidation method
    validation(model, dev_loader, device)
    
    # save model and tokenizer
    #model.save_pretrained("bert_ft")
    #tokenizer.save_pretrained("bert_ft")
    
    
