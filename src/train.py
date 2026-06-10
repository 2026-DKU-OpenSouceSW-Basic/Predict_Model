# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from dataset import BlogDataset
from model import BlogAdClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}")

    # --- [개선] Train/Validation 분리 (8:2) ---
    full_dataset = BlogDataset("output_data.jsonl", fit_stats=True)
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드 고정
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"[INFO] 데이터 분할: 훈련 {train_size}개 / 검증 {val_size}개")

    model = BlogAdClassifier(global_feature_dim=5).to(device)
    
    # --- [개선] ELECTRA 하위 레이어 프리징 (과적합 방지) ---
    # ELECTRA의 12층 중 하위 10층을 고정하고, 상위 2층만 미세조정합니다.
    # 약 1,900개 데이터로 1.1억 파라미터 전체를 학습하면 과적합이 심하기 때문입니다.
    for param in model.electra.embeddings.parameters():
        param.requires_grad = False
    for layer in model.electra.encoder.layer[:10]:  # 12층 중 하위 10층 고정
        for param in layer.parameters():
            param.requires_grad = False
    
    # 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    print(f"[INFO] 파라미터: 전체 {total_params:,}개 / 학습 {trainable_params:,}개 / 고정 {frozen_params:,}개")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 학습 가능한 파라미터만
        lr=2e-5
    )

    epochs = 5
    best_val_f1 = 0.0  # 최고 성능 기록용
    print("[INFO] Start Training...")
    
    for epoch in range(epochs):
        # ==================== 훈련 단계 ====================
        model.train()
        total_loss = 0
        
        # 이번 에폭(Epoch)의 정답과 예측값을 모아둘 리스트
        all_labels = []
        all_preds = []
        
        for batch in train_loader:
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
            
            # --- [정확도 계산을 위해 예측값 수집] ---
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())
            
        # --- [훈련 에폭 평가지표] ---
        avg_loss = total_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # ==================== 검증 단계 ====================
        model.eval()
        val_labels = []
        val_preds = []
        val_probs_list = []  # Brier Score 계산을 위한 raw 확률 수집
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                local_features = batch['local_features'].to(device)
                global_features = batch['global_features'].to(device)
                labels = batch['label'].to(device)

                mask = labels.squeeze() != -1.0
                if not mask.any():
                    continue
                
                input_ids = input_ids[mask]
                attention_mask = attention_mask[mask]
                local_features = local_features[mask]
                global_features = global_features[mask]
                labels = labels[mask]
                
                logits = model(input_ids, attention_mask, local_features, global_features)
                val_loss += criterion(logits, labels).item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs_list.extend(probs.cpu().numpy())  # raw 확률 저장
        
        # --- [검증 평가지표] ---
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # --- [Brier Score] 확률 캘리브레이션 품질 측정 (낮을수록 좋음) ---
        val_labels_flat = np.array(val_labels).flatten()
        val_probs_flat = np.array(val_probs_list).flatten()
        val_brier = brier_score_loss(val_labels_flat, val_probs_flat)
        
        print("=" * 60)
        print(f"🔄 Epoch {epoch+1}/{epochs} 완료!")
        print(f"  [훈련] 📉 Loss: {avg_loss:.4f} | 🎯 Acc: {train_acc*100:.1f}% | 🏆 F1: {train_f1*100:.1f}%")
        print(f"  [검증] 📉 Loss: {avg_val_loss:.4f} | 🎯 Acc: {val_acc*100:.1f}% | 🏆 F1: {val_f1*100:.1f}%")
        print(f"         🔎 Precision: {val_precision*100:.1f}% | 🎣 Recall: {val_recall*100:.1f}%")
        print(f"         📐 Brier Score: {val_brier:.4f} (낮을수록 좋음, 완벽=0.0)")
        
        # 과적합 경고: 훈련 성능은 높은데 검증 성능이 낮으면 경고
        if train_acc - val_acc > 0.15:
            print(f"  ⚠️  과적합 의심! (훈련 Acc {train_acc*100:.1f}% vs 검증 Acc {val_acc*100:.1f}%)")
        
        # 최고 성능 모델 저장 (검증 F1 기준)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "blog_ad_model.pth")
            print(f"  💾 최고 성능 모델 저장! (검증 F1: {val_f1*100:.1f}%)")
        
        print("=" * 60)

    print(f"\n[INFO] 🎉 학습 완료! 최종 최고 검증 F1: {best_val_f1*100:.1f}%")
    print(f"[INFO] 모델 저장됨: blog_ad_model.pth")

if __name__ == "__main__":
    train()