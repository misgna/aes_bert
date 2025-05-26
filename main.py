import pandas as pd
from utils import load_data, EssayDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, BertTokenizer, BertConfig
from bert_model import BERTRegressor
from custom_dataset import CustomDataset

def load_data(filepath):
    return pd.read_csv(filepath)

if __name__ == "__main__":
    train_df = load_data()
    dev_df = load_data()
    
    EPOCHS = 10
    LR = 2e-5 # learning rate
    BATCH_SIZE = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_df, tokenizer)
    val_dataset = CustomDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BERTRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss = F.mse_loss()

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss(outputs, labels)
            loss.backward()
            optimizer.step()
    
        print(f"Epoch {epoch+1}, Training Loss: {loss.item():.4f}")
        
    
