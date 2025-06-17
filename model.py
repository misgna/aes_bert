from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn

class BERTRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size + 25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.pooler_output#outputs.last_hidden_state[:, 0, :]  # CLS token
        combined = torch.cat((cls, features), dim=1)
        cls_dropout = self.dropout(combined)
        output = self.fc(cls_dropout).squeeze(1)
        return self.sigmoid(output)  # shape: (batch_size,)
