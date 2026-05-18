from kiwipiepy import Kiwi
from collections import Counter
import re
import json

class BlogPreprocessor:
    def __init__(self, min_target_count=3):
        print("[INFO] BlogPreprocessor 초기화 중 (Kiwi 로딩)...")
        self.kiwi = Kiwi()
        self.min_target_count = min_target_count
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

    def _find_target_keyword(self, title, body):
        """제목과 본문의 교차 검증을 통해 타겟 키워드(광고 대상)를 찾아냅니다."""
        phrase_counter = self._extract_noun_phrases(body)
        title_no_space = title.replace(" ", "")

        for phrase, count in phrase_counter.most_common():
            if count >= self.min_target_count:

                if phrase in title_no_space:
                    return phrase

                tokens = [t.form for t in self.kiwi.tokenize(phrase) if t.tag in ['NNP', 'NNG']]
                if len(tokens) >= 2 and all(t in title_no_space for t in tokens):
                    return phrase
        return None

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
        target_keyword = self._find_target_keyword(title, body)
        phrase_counter = self._extract_noun_phrases(body)
        
        masked_body = body
        if target_keyword:
            fuzzy_pattern = self._get_fuzzy_pattern(target_keyword)
            masked_body = re.sub(fuzzy_pattern, "[TARGET]", body)

        parts = re.split(r'(\[이미지\])', masked_body)
        
        processed_paragraphs = []
        current_text = ""
        current_images = 0
        
        for part in parts:
            if part == "[이미지]":
                current_images += 1
            else:
                if current_images > 0:
                    processed_paragraphs.append({
                        "text": current_text.strip(),
                        "images": current_images
                    })
                    current_text = part
                    current_images = 0
                else:
                    current_text += part
        
        if current_text.strip() or current_images > 0:
            processed_paragraphs.append({
                "text": current_text.strip(),
                "images": current_images
            })

        total_images = body.count('[이미지]')
        total_len = len(body.replace(" ", ""))
        
        target_frequency = masked_body.count("[TARGET]")
        
        result = {
            "metadata": {
                "title": title,
                "target": target_keyword,
                "label": label
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
