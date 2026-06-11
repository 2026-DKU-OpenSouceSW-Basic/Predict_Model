import torch
import torch.nn as nn
from transformers import AutoModel

class BlogAdClassifier(nn.Module):
    def __init__(self, hidden_dim=256, local_feature_dim=1, global_feature_dim=5,
                 global_proj_dim=64, local_proj_dim=16):
        super(BlogAdClassifier, self).__init__()

        self.electra = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        # dataset.build_tokenizer가 [TARGET] 특수 토큰 1개를 추가하므로 임베딩을 1칸 확장.
        # 새 토큰 행은 기존 임베딩 평균으로 초기화해(무작위 대신) 안정적인 출발점을 준다.
        _old_vocab = self.electra.config.vocab_size
        self.electra.resize_token_embeddings(_old_vocab + 1)
        with torch.no_grad():
            _emb = self.electra.embeddings.word_embeddings.weight
            _emb[_old_vocab] = _emb[:_old_vocab].mean(dim=0)

        self.electra_dim = self.electra.config.hidden_size

        # Local Feature Projection: 문단별 이미지수(1차원)를 16차원으로 확장.
        # 768차원 임베딩에 1차원을 그대로 붙이면 비중이 ~0.1%로 묻히므로,
        # 글로벌 피처와 동일한 취지로 차원을 올려 LSTM이 활용할 여지를 준다.
        self.local_proj = nn.Sequential(
            nn.Linear(local_feature_dim, local_proj_dim),
            nn.ReLU(),
        )

        lstm_input_dim = self.electra_dim + local_proj_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Global Feature Projection: 5차원 → 64차원으로 확장
        # Bi-LSTM 512차원과 대등한 비중으로 반영되도록 차원을 올림
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, global_proj_dim),
            nn.ReLU(),
        )
        
        # 3. 분류기 (Classifier)
        # 기존: (512 + 5) = 517 → GF 비중 1%
        # 변경: (512 + 64) = 576 → GF 비중 11%
        classifier_input_dim = (hidden_dim * 2) + global_proj_dim
        
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

        # 문단별 로컬 피처(이미지수)를 projection으로 확장 후 결합
        local_projected = self.local_proj(local_features)
        lstm_input = torch.cat([para_embeddings, local_projected], dim=-1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        final_hidden_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        # Global Features를 프로젝션 레이어로 확장 후 결합
        global_projected = self.global_proj(global_features)
        classifier_input = torch.cat([final_hidden_state, global_projected], dim=-1)
        
        logits = self.classifier(classifier_input)
        return logits