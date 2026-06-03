import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class BlogDataset(Dataset):
    def __init__(self, json_file, max_paragraphs=10, max_seq_len=128):
        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
            
        self.max_paragraphs = max_paragraphs
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        paragraphs = item.get('paragraphs', [])
        
        # --- Head + Tail Truncation ---
        if len(paragraphs) > self.max_paragraphs:
            head_len = self.max_paragraphs // 2
            tail_len = self.max_paragraphs - head_len
            paragraphs = paragraphs[:head_len] + paragraphs[-tail_len:]
        
        input_ids_list = []
        attention_mask_list = []
        local_features_list = [] 

        for para in paragraphs:
            text = para.get('text', '')
            img_count = para.get('image_count', 0)
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoded['input_ids'].squeeze(0))
            attention_mask_list.append(encoded['attention_mask'].squeeze(0))
            local_features_list.append([img_count])
        
        while len(input_ids_list) < self.max_paragraphs:
            input_ids_list.append(torch.zeros(self.max_seq_len, dtype=torch.long))
            attention_mask_list.append(torch.zeros(self.max_seq_len, dtype=torch.long))
            local_features_list.append([0])
                
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        local_features = torch.tensor(local_features_list, dtype=torch.float)

        # --- 글로벌 피처 추출 (감성점수 포함 총 5개) ---
        gf = item.get('global_features', {})
        global_features = torch.tensor([
            gf.get('total_images', 0) / 100.0,
            gf.get('total_length', 0) / 8000.0,
            gf.get('paragraph_count', 0) / 50.0,
            gf.get('target_frequency', 0) / 40.0,
            gf.get('sentiment_score', 0.5)  # 5번째 피처 (감성점수)
        ], dtype=torch.float)

        label = item['metadata'].get('label')
        label_tensor = torch.tensor([label], dtype=torch.float) if label is not None else torch.tensor([-1.0])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'local_features': local_features,
            'global_features': global_features,
            'label': label_tensor
        }

if __name__ == "__main__":
    dataset = BlogDataset("output_data.jsonl")
    print(f"Dataset length: {len(dataset)}")