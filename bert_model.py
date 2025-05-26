from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn

class BERTRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(cls_output)
        return self.out(x).squeeze(-1)  # shape: (batch_size,)
