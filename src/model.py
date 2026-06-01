import torch
import torch.nn as nn
from transformers import AutoModel

class BlogAdClassifier(nn.Module):
    def __init__(self, hidden_dim=256, local_feature_dim=1, global_feature_dim=5):
        super(BlogAdClassifier, self).__init__()
        
        self.electra = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.electra_dim = self.electra.config.hidden_size 
        
        lstm_input_dim = self.electra_dim + local_feature_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. 분류기 (Classifier)
        classifier_input_dim = (hidden_dim * 2) + global_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, local_features, global_features):
        batch_size, num_para, seq_len = input_ids.size()
        
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        electra_outputs = self.electra(flat_input_ids, attention_mask=flat_attention_mask)
        cls_embeddings = electra_outputs.last_hidden_state[:, 0, :]
        
        para_embeddings = cls_embeddings.view(batch_size, num_para, self.electra_dim)
        
        lstm_input = torch.cat([para_embeddings, local_features], dim=-1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        final_hidden_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        # 텍스트 문맥(Bi-LSTM)과 5개의 보조 지표(Global Features) 결합
        classifier_input = torch.cat([final_hidden_state, global_features], dim=-1)
        
        logits = self.classifier(classifier_input)
        return logits