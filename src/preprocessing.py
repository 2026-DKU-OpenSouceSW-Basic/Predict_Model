from kiwipiepy import Kiwi
import re
import json
import random
from transformers import pipeline

class BlogPreprocessor:
    def __init__(self, min_target_count=3):
        print("[INFO] BlogPreprocessor 초기화 중 (Kiwi 로딩)...")
        self.kiwi = Kiwi()
        self.min_target_count = min_target_count
        
        print("[INFO] 감성 분석 모델 다운로드 및 로딩 중 (최초 1회 수 분 소요)...")
        self.sentiment_analyzer = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)
        print("[OK] 준비 완료!")

    def _extract_noun_phrases(self, text):
        """본문에서 고유명사(NNP)가 포함된 명사구 덩어리를 추출합니다."""
        tokens = self.kiwi.tokenize(text)
        phrases = []
        temp_forms = []
        temp_tags = []

        for token in tokens:
            if token.tag in ['NNP', 'NNG']:
                temp_forms.append(token.form)
                temp_tags.append(token.tag)
            else:
                if temp_forms:
                    if 'NNP' in temp_tags:
                        phrases.append("".join(temp_forms))
                    temp_forms, temp_tags = [], []
        
        if temp_forms and 'NNP' in temp_tags:
            phrases.append("".join(temp_forms))
        return phrases

    def process(self, title, body, label=None, is_my_money=False, detected_reason=None):
        # 1. 전체 이미지 개수 카운트
        total_images = len(re.findall(r'\[image_\d+\]', body, flags=re.IGNORECASE))
        
        # 2. 타겟 키워드 추출 (제목에서)
        target_keywords = self._extract_noun_phrases(title)
        
        # 3. 문단 분리 및 로컬 피처 추출
        paragraphs = [p.strip() for p in body.split('\n') if p.strip()]
        processed_paragraphs = []
        target_frequency = 0
        total_len = len(body)
        
        for para in paragraphs:
            img_count = len(re.findall(r'\[image_\d+\]', para, flags=re.IGNORECASE))
            clean_text = re.sub(r'\[image_\d+\]', '', para).strip()
            clean_text = re.sub(r'\[이미지 텍스트:.*?\]', '', clean_text).strip()
            
            if clean_text:
                for kw in target_keywords:
                    target_frequency += clean_text.count(kw)
                    
                processed_paragraphs.append({
                    "text": clean_text,
                    "image_count": img_count
                })
        
        # 4. 감성 분석 (Soft Label 대신 Sentiment Score 피처로 추출)
        sentiment_score = 0.5
        if processed_paragraphs:
            middle_paras = processed_paragraphs
            num_samples = min(5, len(middle_paras))
            sampled_paras = random.sample(middle_paras, num_samples)
            
            total_negative_score = 0.0
            valid_samples = 0
            
            for para in sampled_paras:
                sample_text = para["text"][:500] 
                if sample_text.strip():
                    sentiment_result = self.sentiment_analyzer(sample_text)[0]
                    star_rating = int(sentiment_result['label'].split()[0]) # 1 ~ 5
                    
                    negative_score = (5 - star_rating) / 4.0
                    total_negative_score += negative_score
                    valid_samples += 1
            
            if valid_samples > 0:
                avg_negative_score = total_negative_score / valid_samples
                sentiment_score = 1.0 - (avg_negative_score * 0.5)
                sentiment_score = round(sentiment_score, 3)

        # 5. 최종 딕셔너리 생성 (오답노트 기록 보존, 정답 보존)
        result = {
            "metadata": {
                "title": title,
                "target": target_keywords,
                "label": label,                                      
                "is_my_money": is_my_money,                          # 모델 입력X
                "detected_reason": detected_reason if detected_reason else [] # 모델 입력X
            },
            "global_features": {
                "total_images": total_images,
                "total_length": total_len,
                "paragraph_count": len(processed_paragraphs),
                "target_frequency": target_frequency,
                "sentiment_score": sentiment_score          # 감성점수를 5번째 피처로 추가
            },
            "paragraphs": processed_paragraphs
        }
        return result