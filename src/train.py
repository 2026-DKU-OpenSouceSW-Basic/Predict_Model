import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import BlogDataset
from model import BlogAdClassifier

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}")

    dataset = BlogDataset("output_data.jsonl")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BlogAdClassifier().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    print("[INFO] Start Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            local_features = batch['local_features'].to(device)
            global_features = batch['global_features'].to(device)
            labels = batch['label'].to(device)

            # 정답 라벨이 비어있는 데이터(-1.0)는 학습에서 제외하는 안전 로직
            mask = labels.squeeze() != -1.0
            if not mask.any():
                continue
            
            input_ids = input_ids[mask]
            attention_mask = attention_mask[mask]
            local_features = local_features[mask]
            global_features = global_features[mask]
            labels = labels[mask]
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask, local_features, global_features)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()