import torch
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def predict(dataset):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    qwk = cohen_kappa_score(trues, preds, weights="quadratic")
    print(f"Validation RMSE: {qwk:.4f}")

if __name__ == "__main__":
    # load model 
    model_name = "bert_ft"
    model =BertForSequenceClassification.from_pretrained("bert_ft")
    tokenizer = BertTokenizer.from_pretrained("bert_ft")
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    # data
    test = pd.read_csv("datasets/XLEAF/test.csv")
    