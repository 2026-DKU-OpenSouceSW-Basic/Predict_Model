import torch
import torch.nn as nn
from transformers import AutoModel

class BlogAdClassifier(nn.Module):
    def __init__(self, hidden_dim=256, local_feature_dim=1, global_feature_dim=4):
        super(BlogAdClassifier, self).__init__()
        
        # 1. kcELECTRA 모델 로드 (문단 임베딩 추출용)
        self.electra = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        
        # ELECTRA의 출력 차원 (기본 모델은 보통 768)
        self.electra_dim = self.electra.config.hidden_size 
        
        # 2. Bi-LSTM 계층 (문단의 순서/문맥 학습)
        # 입력: electra_dim (768) + 문단별 이미지 개수 (1) = 769
        lstm_input_dim = self.electra_dim + local_feature_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. 분류기 (Classifier)
        # Bi-LSTM 출력 (hidden_dim * 2) + 글로벌 피처(4)
        classifier_input_dim = (hidden_dim * 2) + global_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # 이진 분류 (0: 정상, 1: 광고)
        )

    def forward(self, input_ids, attention_mask, local_features, global_features):
        """
        input_ids: (batch_size, max_paragraphs, max_seq_len)
        """
        batch_size, num_para, seq_len = input_ids.size()
        
        # ELECTRA에 넣기 위해 Batch와 Paragraph 차원을 하나로 합침
        # -> (batch_size * max_paragraphs, max_seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        # ELECTRA 통과
        electra_outputs = self.electra(flat_input_ids, attention_mask=flat_attention_mask)
        
        # [CLS] 토큰 벡터만 추출하여 문단 임베딩으로 사용 -> (batch_size * num_para, 768)
        cls_embeddings = electra_outputs.last_hidden_state[:, 0, :]
        
        # 원래의 (batch_size, num_para, 768) 형태로 복구
        para_embeddings = cls_embeddings.view(batch_size, num_para, self.electra_dim)
        
        # 문단별 로컬 피처(이미지 수) 결합 -> (batch_size, num_para, 769)
        lstm_input = torch.cat([para_embeddings, local_features], dim=-1)
        
        # Bi-LSTM 통과
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # 마지막 타임스텝의 양방향 Hidden state 추출 -> (batch_size, hidden_dim * 2)
        # h_n shape: (num_directions * num_layers, batch_size, hidden_size)
        final_hidden_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        # 글로벌 피처(전체 이미지, 길이 등)와 결합 -> (batch_size, hidden_dim * 2 + 4)
        final_representation = torch.cat([final_hidden_state, global_features], dim=-1)
        
        # 분류기 통과 -> (batch_size, 1)
        logits = self.classifier(final_representation)
        
        return logits
