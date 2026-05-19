from kiwipiepy import Kiwi
from collections import Counter
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
        
        return Counter(phrases)

    def _find_target_keywords(self, title, body):
        """제목과 본문의 교차 검증을 통해 타겟 키워드(광고 대상)를 모두 찾아냅니다."""
        phrase_counter = self._extract_noun_phrases(body)
        title_no_space = title.replace(" ", "")

        targets = []
        for phrase, count in phrase_counter.most_common():
            if count >= self.min_target_count:

                if phrase in title_no_space:
                    targets.append(phrase)
                    continue

                tokens = [t.form for t in self.kiwi.tokenize(phrase) if t.tag in ['NNP', 'NNG']]
                if len(tokens) >= 2 and all(t in title_no_space for t in tokens):
                    targets.append(phrase)
        
        return targets if targets else None

    def _get_fuzzy_pattern(self, keyword):
        """키워드의 핵심 명사들 사이에 임의의 문자가 올 수 있는 정규표현식을 생성합니다."""
        tokens = [t.form for t in self.kiwi.tokenize(keyword) if t.tag in ['NNP', 'NNG']]
        
        if not tokens:
            return re.escape(keyword)
        
        if len(tokens) == 1:
            return r'\s*'.join(list(re.escape(tokens[0])))
        
        pattern = r'\s*[^.!?]{0,10}\s*'.join([re.escape(t) for t in tokens])
        return pattern

    def process(self, title, body, label=None):
        """
        크롤링된 원본 데이터를 받아 학습용 정제 데이터(JSON 형식)로 변환합니다.
        [이미지] 태그를 기준으로 문단을 나누는 새로운 로직을 적용합니다.
        """
        target_keywords = self._find_target_keywords(title, body)
        phrase_counter = self._extract_noun_phrases(body)
        
        masked_body = body
        if target_keywords:
            target_keywords.sort(key=len, reverse=True)
            for keyword in target_keywords:
                fuzzy_pattern = self._get_fuzzy_pattern(keyword)
                masked_body = re.sub(fuzzy_pattern, "[TARGET]", masked_body)

        parts = re.split(r'(\[이미지\])', masked_body)
        
        processed_paragraphs = []
        current_text = ""
        current_images = 0
        
        for part in parts:
            if part == "[이미지]":
                current_images += 1
            else:
                current_text += part + " "
                

                if len(current_text.replace(" ", "")) >= 50:
                    processed_paragraphs.append({
                        "text": current_text.strip(),
                        "images": current_images
                    })
                    current_text = ""
                    current_images = 0
        
        if current_text.strip() or current_images > 0:
            processed_paragraphs.append({
                "text": current_text.strip(),
                "images": current_images
            })

        total_images = body.count('[이미지]')
        total_len = len(body.replace(" ", ""))
        
        target_frequency = masked_body.count("[TARGET]")
        
        soft_label = label
        if label == 1 and processed_paragraphs:
            num_samples = min(5, len(processed_paragraphs))
            sampled_paras = random.sample(processed_paragraphs, num_samples)
            
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
                soft_label = 1.0 - (avg_negative_score * 0.5)
                soft_label = round(soft_label, 3)

        result = {
            "metadata": {
                "title": title,
                "target": target_keywords,
                "label": soft_label
            },
            "global_features": {
                "total_images": total_images,
                "total_length": total_len,
                "paragraph_count": len(processed_paragraphs),
                "target_frequency": target_frequency
            },
            "paragraphs": processed_paragraphs
        }
        
        return result
