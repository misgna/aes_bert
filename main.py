import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_data
from model import BertRegressionModel
from custom_dataset import EssayDataset

def set_seed(seed):
    # Python built-in random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def training(model, train_dl, optimizer, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            print(batch['input_ids'])

            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(outputs, batch['score'])
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

def validation():
    pass

if __name__  == '__main__':
    set_seed(42)
    prompt = 1
    
    asap_folds = load_data(prompt)
    for fold in asap_folds:
        train, val, test = fold
        
        train_ds = EssayDataset(train['essay'].to_list(), train['score'].to_list())
        val_ds = EssayDataset(val['essay'].to_list(), val['score'].to_list())
        test_ds = EssayDataset(test['essay'].to_list(), test['score'].to_list())

        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)

        model = BertRegressionModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = nn.MSELoss()  # or L1Loss for Mean Absolute Error
        epochs = 5

        training(model, train_dl, optimizer, loss_fn, epochs)
        
    
