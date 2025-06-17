import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import BERTRegressor
from custom_dataset import CustomDataset
from utils import load_asap_data, load_asap_data_cross, load_leaf_data, evaluate_qwk

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def training(model, dataloader, optimizer, loss_fn, epochs, desc):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=desc):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            features = batch['features'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Training Loss: {total_loss}")

def validation(model, dataloader, desc):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            features = batch['features'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            preds = outputs.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description="BERT-based holistic AES")
    parser.add_argument('--dataset', type=str, help='For dataset input asap (asap for prompt-specific), cross(asap for cross-prompt) or leaf')
    parser.add_argument('--prompt', type=int, default=1, help='Choose from 1-8 for asap or 9 for leaf')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    
    return args.dataset, args.prompt, args.batch_size, args.epochs

if __name__ == "__main__":
    seed_list = [12, 22, 32, 42, 52]
    set_seed(42)
    
    dataset, prompt, batch_size, epochs = main()
    loss_fn = nn.MSELoss()
    learning_rate = 2e-5 # learning rate
    
    data_folds = []
    
    if dataset == 'asap':
        data_folds = load_asap_data(prompt)
    elif dataset == 'cross':
        data_folds = load_asap_data_cross(prompt)
    elif dataset == 'leaf':
        prompt = 9
        data_folds = load_leaf_data(prompt)    

    test_qwks = []
    
    for fold in data_folds:
        model = BERTRegressor().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train, val, test = fold
        
        train_ds = CustomDataset(train['essay'].to_list(), train['score'].to_list(), train['features'])
        val_ds = CustomDataset(val['essay'].to_list(), val['score'].to_list(), val['features'])
        test_ds = CustomDataset(test['essay'].to_list(), test['score'].to_list(), test['features'])
        
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        training(model, train_dl,  optimizer, loss_fn, epochs, "Training")
        
        labels, preds = validation(model, val_dl, "Validation")
        
        qwk = evaluate_qwk(preds, labels, prompt)
        print(f'QWK: {qwk}')
        
        # Collect test results
        labels_test, preds_test = validation(model, test_dl, "Testing")
        qwk_test = evaluate_qwk(preds_test, labels_test, prompt)
        test_qwks.append(qwk_test)
        
    print('--' * 10)
    for (index, qwk) in enumerate(test_qwks):
        print(f'Fold {index}: {qwk}')
        
    print(f'Avg QWK: {sum(test_qwks) / len(test_qwks)}')
    
    
