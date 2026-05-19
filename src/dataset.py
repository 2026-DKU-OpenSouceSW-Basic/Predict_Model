import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class BlogDataset(Dataset):
    def __init__(self, json_file, max_paragraphs=10, max_seq_len=128):
        # 1. 데이터 로드 (JSONL 형식 읽기)
        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
            
        self.max_paragraphs = max_paragraphs
        self.max_seq_len = max_seq_len
        # kcELECTRA 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        paragraphs = item['paragraphs']
        
        # --- Head + Tail Truncation ---
        if len(paragraphs) > self.max_paragraphs:
            head_len = self.max_paragraphs // 2
            tail_len = self.max_paragraphs - head_len
            paragraphs = paragraphs[:head_len] + paragraphs[-tail_len:]
        
        input_ids_list = []
        attention_mask_list = []
        local_features_list = [] # 문단별 이미지 개수 등

        # 문단 순회하며 토큰화
        for para in paragraphs:
            text = para.get('text', '')
            images = para.get('images', 0)
            
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoded['input_ids'].squeeze(0))
            attention_mask_list.append(encoded['attention_mask'].squeeze(0))
            local_features_list.append([images])

        # 문단 개수가 max_paragraphs보다 적으면 Padding 추가
        pad_len = self.max_paragraphs - len(paragraphs)
        if pad_len > 0:
            for _ in range(pad_len):
                input_ids_list.append(torch.zeros(self.max_seq_len, dtype=torch.long))
                attention_mask_list.append(torch.zeros(self.max_seq_len, dtype=torch.long))
                local_features_list.append([0])
                
        # 텐서로 변환 -> Shape: (max_paragraphs, max_seq_len)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        local_features = torch.tensor(local_features_list, dtype=torch.float)

        # --- 글로벌 피처 추출 ---
        gf = item.get('global_features', {})
        global_features = torch.tensor([
            gf.get('total_images', 0),
            gf.get('total_length', 0),
            gf.get('paragraph_count', 0),
            gf.get('target_frequency', 0)
        ], dtype=torch.float)

        # 라벨 (있으면 float 텐서로, 없으면 -1)
        label = item['metadata'].get('label')
        label_tensor = torch.tensor([label], dtype=torch.float) if label is not None else torch.tensor([-1.0])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'local_features': local_features,
            'global_features': global_features,
            'label': label_tensor
        }

# 테스트 코드 (이 파일만 실행했을 때 작동)
if __name__ == "__main__":
    dataset = BlogDataset("dummy_data.jsonl")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Input IDs Shape:", batch['input_ids'].shape) # (Batch, Para, Seq)
        print("Labels:", batch['label'])
        break
