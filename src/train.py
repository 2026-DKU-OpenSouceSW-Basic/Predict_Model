import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import BlogDataset
from model import BlogAdClassifier
import numpy as np

def train():
    # 1. 설정 및 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}")

    # 데이터 로더 (미리 만든 더미 데이터 사용)
    dataset = BlogDataset("dummy_data.jsonl")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 모델 초기화
    model = BlogAdClassifier().to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    print("[INFO] Start Training...")
    
    # 2. 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # 데이터를 Device(GPU/CPU)로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            local_features = batch['local_features'].to(device)
            global_features = batch['global_features'].to(device)
            labels = batch['label'].to(device)

            # 라벨이 없는 데이터(-1.0)는 건너뛰거나 제외하는 로직이 필요 (여기서는 패스)
            
            optimizer.zero_grad()
            
            # Forward Pass (예측)
            logits = model(input_ids, attention_mask, local_features, global_features)
            
            # Loss Calculation (오차 계산)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Backward Pass (역전파) & Weight Update
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("[OK] Training Completed!")

if __name__ == "__main__":
    train()
