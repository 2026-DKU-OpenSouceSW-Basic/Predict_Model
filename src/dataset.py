import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import os

MODEL_NAME = "beomi/KcELECTRA-base-v2022"
TARGET_TOKEN = "[TARGET]"  # 제목 키워드 치환용 특수 토큰 (단일 토큰으로 등록)

def build_tokenizer():
    """[TARGET] 특수 토큰이 등록된 토크나이저를 생성한다.

    이걸 거치지 않으면 '[TARGET]'이 ['[','T','##AR','##G','##ET',']'] 6개로
    쪼개져 무의미해진다. 학습/추론이 동일하게 이 헬퍼를 써야 토큰 id가 맞는다.
    여기서 정확히 1개 토큰을 추가하므로, model.py는 임베딩을 +1 확장한다.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': [TARGET_TOKEN]})
    return tokenizer

DEFAULT_FEATURE_STATS = {
    'total_images': {'mean': 26.1210, 'std': 17.0855},
    'total_length': {'mean': 2613.5765, 'std': 990.7447},
    'paragraph_count': {'mean': 13.5631, 'std': 5.2045},
    'target_frequency': {'mean': 2.5796, 'std': 6.2931},
    'sentiment_score': {'mean': 0.7143, 'std': 0.0855},
}

_cached_stats = None

def get_feature_stats():
    global _cached_stats
    if _cached_stats is not None:
        return _cached_stats
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_dir, "feature_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                _cached_stats = json.load(f)
            return _cached_stats
        except Exception as e:
            print(f"[WARNING] Failed to load {stats_path}: {e}")
            
    _cached_stats = DEFAULT_FEATURE_STATS
    return _cached_stats

def normalize_global_features(gf):
    stats = get_feature_stats()
    
    if isinstance(gf, dict):
        total_images = gf.get('total_images', 0)
        total_length = gf.get('total_length', 0)
        paragraph_count = gf.get('paragraph_count', 0)
        target_frequency = gf.get('target_frequency', 0)
        sentiment_score = gf.get('sentiment_score', 0.5)
    else:
        total_images = gf['total_images']
        total_length = gf['total_length']
        paragraph_count = gf['paragraph_count']
        target_frequency = gf['target_frequency']
        sentiment_score = gf['sentiment_score']
        
    norm_images = (total_images - stats['total_images']['mean']) / stats['total_images']['std']
    norm_length = (total_length - stats['total_length']['mean']) / stats['total_length']['std']
    norm_paragraphs = (paragraph_count - stats['paragraph_count']['mean']) / stats['paragraph_count']['std']
    norm_frequency = (target_frequency - stats['target_frequency']['mean']) / stats['target_frequency']['std']

    # 감성 점수도 다른 4개와 동일하게 z-score로 정규화하여 같은 스케일로 학습에 반영.
    # (기존엔 0~1 raw로 들어가 스케일이 달라 상대적으로 묻혔음)
    # 구버전 feature_stats.json에 sentiment가 없을 수 있어 안전한 기본값으로 fallback.
    sent_stats = stats.get('sentiment_score', DEFAULT_FEATURE_STATS['sentiment_score'])
    norm_sentiment = (sentiment_score - sent_stats['mean']) / sent_stats['std']

    return [norm_images, norm_length, norm_paragraphs, norm_frequency, norm_sentiment]

class BlogDataset(Dataset):
    def __init__(self, json_file, max_paragraphs=10, max_seq_len=128, fit_stats=False, stats_file="feature_stats.json"):
        global _cached_stats
        
        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
            
        self.max_paragraphs = max_paragraphs
        self.max_seq_len = max_seq_len
        self.tokenizer = build_tokenizer()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.stats_path = os.path.join(current_dir, stats_file)
        
        if fit_stats:
            self.stats = self._calculate_stats()
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=4)
            _cached_stats = self.stats
            print(f"[INFO] Calculated and saved feature stats to {self.stats_path}")
        else:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
                _cached_stats = self.stats
                print(f"[INFO] Loaded feature stats from {self.stats_path}")
            else:
                print(f"[WARNING] {self.stats_path} not found. Calculating stats dynamically from dataset...")
                self.stats = self._calculate_stats()
                _cached_stats = self.stats


    def _calculate_stats(self):
        import numpy as np
        imgs, lens, paras, freqs, sents = [], [], [], [], []
        for item in self.data:
            gf = item.get('global_features', {})
            imgs.append(gf.get('total_images', 0))
            lens.append(gf.get('total_length', 0))
            paras.append(gf.get('paragraph_count', 0))
            freqs.append(gf.get('target_frequency', 0))
            sents.append(gf.get('sentiment_score', 0.5))

        std_imgs = np.std(imgs)
        std_lens = np.std(lens)
        std_paras = np.std(paras)
        std_freqs = np.std(freqs)
        std_sents = np.std(sents)

        return {
            'total_images': {'mean': float(np.mean(imgs)), 'std': float(std_imgs if std_imgs > 0 else 1.0)},
            'total_length': {'mean': float(np.mean(lens)), 'std': float(std_lens if std_lens > 0 else 1.0)},
            'paragraph_count': {'mean': float(np.mean(paras)), 'std': float(std_paras if std_paras > 0 else 1.0)},
            'target_frequency': {'mean': float(np.mean(freqs)), 'std': float(std_freqs if std_freqs > 0 else 1.0)},
            'sentiment_score': {'mean': float(np.mean(sents)), 'std': float(std_sents if std_sents > 0 else 1.0)}
        }

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
        norm_gf = normalize_global_features(gf)
        global_features = torch.tensor(norm_gf, dtype=torch.float)

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