from kiwipiepy import Kiwi
import re
from transformers import pipeline

class BlogPreprocessor:
    def __init__(self, min_target_count=3, use_bert_sentiment=False):
        import os
        env_val = os.environ.get("USE_BERT_SENTIMENT")
        if env_val is not None:
            use_bert_sentiment = env_val.lower() in ("true", "1", "yes")

        print("[INFO] BlogPreprocessor 초기화 중 (Kiwi 로딩)...")
        self.kiwi = Kiwi()
        self.min_target_count = min_target_count
        self.use_bert_sentiment = use_bert_sentiment
        self.sentiment_analyzer = None
        print("[OK] 준비 완료!")

    def _remove_ad_disclosure(self, text):
        patterns = [
            r'본\s*포스팅은.*?작성되었습니다\.?',
            r'본\s*게시[글물]은.*?작성되었습니다\.?',
            r'본\s*포스팅은.*?작성하였습니다\.?',
            r'소정의\s*원고료를.*?작성하였습니다\.?',
            r'업체로부터.*?제공받아.*?작성.*?\.?',
            r'이\s*포스팅은.*?일환으로.*?\.?',
            r'대가를\s*제공받아.*?작성.*?\.?',
            r'원고료를\s*지원받아.*?작성.*?\.?',
            r'본\s*콘텐츠는.*?제공받아.*?\.?',
            r'내돈내산\s*(으로|입니다|이에요|후기|리뷰)',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)

        keyword_patterns = [
            r'제공받아\s*작성',
            r'지원받아\s*작성',
            r'지급받아\s*작성',
            r'원고료를?\s*(제공|지원|지급|받아)',
            r'소정의\s*원고료',
            r'경제적\s*대가를?\s*(제공|지원|지급|받아)',
            r'를\s*지원받아\s*작성',
            r'를\s*제공받아\s*작성',
            r'제품\s*협찬',
            r'협찬을?\s*받아',
            r'협찬받았',
            r'체험단',
            r'무상\s*제공',
            r'제공받았습니다',
            r'지원받았습니다',
            r'협찬받았습니다',
        ]
        for pattern in keyword_patterns:
            text = re.sub(pattern, '', text)
        return text

    def _clean_blog_metadata(self, text):
        patterns = [
            r'\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*\d{1,2}:\d{2}',  # 날짜/시간
            r'이웃추가',
            r'본문\s*기타\s*기능',
            r'공감\s*\d+',
            r'댓글\s*\d+',
            r'©\s*NAVER\s*Corp\.?',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

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
        # 0. 광고 표시 문구 및 블로그 메타데이터 제거 (Data Leakage 방지)
        body = self._remove_ad_disclosure(body)
        body = self._clean_blog_metadata(body)

        # 1. 전체 이미지 개수 카운트
        total_images = len(re.findall(r'\[image_\d+\]', body, flags=re.IGNORECASE))

        # 2. 타겟 키워드 추출 (제목에서)
        target_keywords = self._extract_noun_phrases(title)

        # 3. 문단 분리 및 로컬 피처 추출 (150자 단위 병합 로직 적용)
        paragraphs = [p.strip() for p in body.split('\n') if p.strip()]
        processed_paragraphs = []
        target_frequency = 0
        total_len = len(body)

        # 문단을 뭉치기 위한 임시 바구니
        temp_text = ""
        temp_img_count = 0
        MIN_PARA_LENGTH = 150 # 한 문단의 최소 글자 수

        for para in paragraphs:
            img_count = len(re.findall(r'\[image_\d+\]', para, flags=re.IGNORECASE))
            clean_text = re.sub(r'\[image_\d+\]', '', para).strip()
            # OCR 텍스트는 모델 학습/입력에서 제외합니다.
            # 광고 고지 이미지는 ProbabilityFusion에서 별도 확률 소스로 결합합니다.
            clean_text = re.sub(r'\[이미지 텍스트:.*?\]', '', clean_text).strip()

            if clean_text:
                for kw in target_keywords:
                    target_frequency += clean_text.count(kw)
                    clean_text = clean_text.replace(kw, '[TARGET]')

            # 텍스트와 이미지를 임시 바구니에 계속 담습니다
            if clean_text:
                temp_text += clean_text + " "
            temp_img_count += img_count

            # 바구니에 담긴 글자가 150자를 넘어가면 비로소 하나의 '문단'으로 확정!
            if len(temp_text.strip()) >= MIN_PARA_LENGTH:
                processed_paragraphs.append({
                    "text": temp_text.strip(),
                    "image_count": temp_img_count
                })
                # 바구니 초기화
                temp_text = ""
                temp_img_count = 0

        # 반복문이 끝난 후, 바구니에 미처 다 채우지 못한 찌꺼기(마지막 문단)가 남아있다면 털어넣기
        if temp_text.strip() or temp_img_count > 0:
            processed_paragraphs.append({
                "text": temp_text.strip(),
                "image_count": temp_img_count
            })


        # 4. 감성 분석 (Soft Label 대신 Sentiment Score 피처로 추출)
        sentiment_score = 0.5
        if processed_paragraphs:
            if self.use_bert_sentiment:
                if self.sentiment_analyzer is None:
                    print("[INFO] 감성 분석 모델 로딩 중 (최초 1회 수 분 소요)...")
                    try:
                        # PaddleOCR와 PyTorch 간의 CUDA 드라이버 충돌 및 GPU 메모리 부족(OOM) 방지를 위해 감성 분석은 CPU에서 수행하도록 설정합니다.
                        self.sentiment_analyzer = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)
                        print("[OK] 감성 분석 모델 로딩 완료!")
                    except Exception as e:
                        print(f"[WARNING] 감성 분석 모델 로드 실패 (가상 메모리 부족 등으로 인한 로드 실패): {e}")
                        print("[INFO] 감성 분석 모델 대신 기본 감성 점수(0.7)를 사용하는 Fallback 모드로 동작합니다.")
                        self.sentiment_analyzer = "fallback"

                if self.sentiment_analyzer != "fallback":
                    middle_paras = processed_paragraphs
                    num_samples = min(5, len(middle_paras))
                    if num_samples == 0:
                        sampled_paras = []
                    else:
                        step = max(1, len(middle_paras) / num_samples)
                        sampled_paras = [middle_paras[int(i * step)] for i in range(num_samples)]

                    total_negative_score = 0.0
                    valid_samples = 0

                    for para in sampled_paras:
                        sample_text = para["text"][:500]
                        if sample_text.strip():
                            try:
                                sentiment_result = self.sentiment_analyzer(sample_text)[0]
                                star_rating = int(sentiment_result['label'].split()[0]) # 1 ~ 5

                                negative_score = (5 - star_rating) / 4.0
                                total_negative_score += negative_score
                                valid_samples += 1
                            except Exception as e:
                                print(f"[WARNING] 감성 분석 수행 중 에러 발생: {e}")

                    if valid_samples > 0:
                        avg_negative_score = total_negative_score / valid_samples
                        sentiment_score = 1.0 - (avg_negative_score * 0.5)
                        sentiment_score = round(sentiment_score, 3)
                    else:
                        sentiment_score = 0.7
                else:
                    sentiment_score = self._estimate_keyword_sentiment(processed_paragraphs)
            else:
                # 경량 키워드 기반 감성 분석 수행 (메모리 사용량 0MB)
                sentiment_score = self._estimate_keyword_sentiment(processed_paragraphs)

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

    def _estimate_keyword_sentiment(self, processed_paragraphs):
        # 본문 텍스트 병합 (최대 5개 문단 샘플링)
        num_samples = min(5, len(processed_paragraphs))
        if num_samples == 0:
            return 0.75

        step = max(1, len(processed_paragraphs) / num_samples)
        sampled_paras = [processed_paragraphs[int(i * step)] for i in range(num_samples)]
        combined_text = " ".join([p["text"] for p in sampled_paras])

        # 부정/긍정 키워드 카운트
        neg_words = ["아쉽", "아쉬운", "별로", "실망", "단점", "불편", "최악", "우려", "틀린", "부족", "안좋", "어렵"]
        pos_words = ["추천", "좋다", "좋은", "최고", "만족", "감동", "깔끔", "친절", "맛있다", "맛있", "예쁘", "훌륭", "편하"]

        neg_count = sum(combined_text.count(w) for w in neg_words)
        pos_count = sum(combined_text.count(w) for w in pos_words)

        if neg_count + pos_count == 0:
            return 0.75

        neg_ratio = neg_count / (neg_count + pos_count)
        # 부정 비율에 따라 0.5 ~ 1.0 사이의 점수 부여
        score = 1.0 - (neg_ratio * 0.5)
        return round(score, 3)
