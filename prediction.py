from sklearn.metrics import mean_squared_error

model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        preds.extend(outputs.cpu().numpy())
        trues.extend(labels.cpu().numpy())

rmse = mse(trues, preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")
