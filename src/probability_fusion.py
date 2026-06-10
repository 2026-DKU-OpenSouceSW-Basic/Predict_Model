# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
"""
소프트 확률 결합 모듈 (Soft Probability Fusion)

패널티 점수를 산술적으로 더하는 대신, 키워드/OCR 검출 결과를
독립적인 확률 소스로 모델링하고 Log-odds 공간에서 가중 결합합니다.

결합 방식: Logarithmic Opinion Pool
    logit_final = w1 * log(p_model/(1-p_model))
                + w2 * log(p_rule/(1-p_rule))
                + w3 * log(p_ocr/(1-p_ocr))
    p_final = sigmoid(logit_final)
"""
import math
import re


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """확률값을 log-odds로 변환. 0과 1 근처에서의 수치 안정성 보장."""
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """log-odds를 확률로 역변환."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _compact_for_match(text: str) -> str:
    """OCR 띄어쓰기/기호 흔들림을 줄이기 위한 매칭용 정규화."""
    return re.sub(r'[^0-9A-Za-z가-힣]+', '', text or '').lower()


# =========================================================
# 키워드 규칙 정의 (패턴, 소프트 확률, 사유 설명)
# 확률은 "이 패턴이 등장했을 때 광고일 사후 확률"을 의미
# =========================================================
_TEXT_RULES = [
    # --- 강한 광고 증거 (p >= 0.85) ---
    (r'소정의\s*원고료',                 0.92, "소정의 원고료 검출"),
    (r'제공받아\s*작성',                 0.90, "제공받아 작성 검출"),
    (r'원고료를?\s*(지원|제공|지급)',      0.90, "원고료 지원/제공/지급 검출"),
    (r'원고료를?\s*받아',                0.88, "원고료 수령 검출"),
    (r'경제적\s*대가',                  0.88, "경제적 대가 검출"),
    (r'(식사|이용|시식|체험|제품)권을?\s*(제공|지원|지급)받아', 0.88, "식사권/이용권 제공 검출"),
    (r'서비스를?\s*제공받아',             0.86, "서비스 제공 검출"),
    (r'업체로부터.*?제공',               0.85, "업체 제공 검출"),
    (r'무상으로?\s*제공',                0.85, "무상 제공 검출"),
    (r'협찬을?\s*받아',                  0.85, "협찬 지원 검출"),
    (r'협찬(?!은?\s*(?:아닙|아닌|X|x|없음|안\s*받))', 0.85, "협찬 키워드 검출"),
    (r'광고(?!은?\s*(?:아닙|아닌|X|x|없음|안\s*받))', 0.85, "광고 키워드 검출"),

    # --- 중간 강도 증거 (0.70 <= p < 0.85) ---
    (r'제품을?\s*협찬',                 0.82, "제품 협찬 검출"),
    (r'이\s*포스팅은.*?일환',            0.80, "마케팅 일환 검출"),
    (r'체험단',                        0.78, "체험단 검출"),
    (r'시식권|식사권|이용권',             0.75, "식사권/이용권 키워드 검출"),

    # --- 약한 음의 증거 (p < 0.50) ---
    (r'내돈내산',                       0.35, "내돈내산 자기선언"),
]

_OCR_KEYWORDS = [
    # (키워드, 검출 시 확률, 사유)
    ('공정거래위원회', 0.95, "이미지 OCR: '공정거래위원회' 검출"),
    ('대가성',    0.92, "이미지 OCR: '대가성' 검출"),
    ('광고표기',  0.90, "이미지 OCR: '광고표기' 검출"),
    ('원고료',    0.90, "이미지 OCR: '원고료' 검출"),
    ('협찬',      0.85, "이미지 OCR: '협찬' 검출"),
    ('제공받아',  0.88, "이미지 OCR: '제공받아' 검출"),
    ('제공받았습니다', 0.88, "이미지 OCR: '제공받았습니다' 검출"),
    ('지원받아',  0.85, "이미지 OCR: '지원받아' 검출"),
    ('지원받았습니다', 0.85, "이미지 OCR: '지원받았습니다' 검출"),
    ('소정의',    0.88, "이미지 OCR: '소정의' 검출"),
    ('식사권',    0.85, "이미지 OCR: '식사권' 검출"),
    ('이용권',    0.85, "이미지 OCR: '이용권' 검출"),
    ('시식권',    0.85, "이미지 OCR: '시식권' 검출"),
    ('초대권',    0.82, "이미지 OCR: '초대권' 검출"),
    ('방문권',    0.82, "이미지 OCR: '방문권' 검출"),
    ('체험단',    0.80, "이미지 OCR: '체험단' 검출"),
    ('무상',      0.82, "이미지 OCR: '무상' 검출"),
    ('광고',      0.72, "이미지 OCR: '광고' 검출"),
    ('내돈내산',  0.35, "이미지 OCR: '내돈내산' 자기선언"),

    # --- OCR로 읽힌 체험단 플랫폼/공정위 배너 문구 ---
    ('공정위',    0.95, "공정위 대가성 표기 이미지 검출"),
    ('리뷰노트',  0.95, "리뷰노트 대가성 표기 이미지 검출"),
    ('디너의여왕',0.95, "디너의여왕 대가성 표기 이미지 검출"),
    ('미스터블로그',0.95, "미스터블로그 대가성 표기 이미지 검출"),
    ('레뷰',      0.95, "레뷰 대가성 표기 이미지 검출"),
    ('스토리앤',  0.95, "스토리앤 대가성 표기 이미지 검출"),
    ('놀러와체험단',0.95, "놀러와체험단 대가성 표기 이미지 검출"),
    ('서울오빠',  0.95, "서울오빠 대가성 표기 이미지 검출"),
]


# =========================================================
# 체험단/리뷰 플랫폼 이미지 URL 규칙
# 광고 고지 배너는 외부 체험단 플랫폼(reviewnote 등)이나 공정위 고지 경로에서
# 호스팅되는 경우가 많다. 이미지 다운로드/OCR 없이 URL 문자열만으로도 강한
# 광고 증거가 되므로, OCR이 배너를 못 읽거나 선택에서 누락돼도 잡아내는 안전망.
# 확률은 OCR '협찬' 검출과 동일한 강한 증거 등급(0.85)으로 맞춘다.
# =========================================================
_PLATFORM_URL_RULES = [
    ("reviewnote.cloud", 0.85, "리뷰노트 체험단 배너"),
    ("/gongjeong",       0.85, "공정위 광고 표기"),
    ("revu.net",         0.85, "레뷰 체험단 배너"),
    ("dinnerqueen",      0.85, "디너의여왕 체험단 배너"),
    ("mrblog",           0.85, "미스터블로그 체험단 배너"),
    ("seoulouba",        0.85, "서울오빠 체험단 배너"),
]


class ProbabilityFusion:
    """규칙 기반 검출 결과를 소프트 확률로 변환하고 모델 확률과 결합합니다.

    Args:
        w_model: 딥러닝 모델 확률에 대한 가중치 (기본값 1.0)
        w_rule:  텍스트 규칙 확률에 대한 가중치 (기본값 0.6)
        w_ocr:   OCR 검출 확률에 대한 가중치 (기본값 0.5)
    """

    def __init__(self, w_model: float = 1.0, w_rule: float = 0.6, w_ocr: float = 0.5, w_url: float = 0.6,
                 min_ad_prob: float = 0.55):
        self.w_model = w_model
        self.w_rule = w_rule
        self.w_ocr = w_ocr
        self.w_url = w_url
        # 강한 법적 증거가 있을 때 보장할 최소 광고 확률.
        # 결정 임계치(0.5)를 확실히 넘기기 위한 값으로, 85% 고정 대신 사용한다.
        self.min_ad_prob = min_ad_prob

    def keyword_to_probability(self, raw_content: str) -> tuple[float, list[str]]:
        """본문 텍스트에서 규칙 기반 광고 확률을 산출합니다.

        여러 규칙이 매칭되면, 각 규칙의 log-odds를 합산하여
        단일 확률로 통합합니다 (Naive Bayes 스타일 결합).

        Args:
            raw_content: 전처리 전 원본 본문 텍스트

        Returns:
            (p_rule, reasons): 결합된 규칙 확률과 검출 사유 리스트
        """
        if not raw_content:
            return 0.5, []

        matched_probs = []
        reasons = []

        for pattern, prob, reason in _TEXT_RULES:
            if re.search(pattern, raw_content):
                matched_probs.append(prob)
                reasons.append(f"{reason} (p={prob:.0%})")

        if not matched_probs:
            return 0.5, []  # 중립 (prior)

        # 복수 규칙 매칭 시: log-odds 합산 후 sigmoid
        # prior = 0.5 (logit = 0)이므로, 각 규칙의 logit을 그대로 합산
        combined_logit = sum(_safe_logit(p) for p in matched_probs)
        p_rule = _sigmoid(combined_logit)

        return p_rule, reasons

    def ocr_to_probability(self, ocr_texts: list[str]) -> tuple[float, list[str]]:
        """OCR로 추출된 이미지 텍스트에서 광고 확률을 산출합니다.

        Args:
            ocr_texts: OCR로 추출된 텍스트 리스트

        Returns:
            (p_ocr, reasons): OCR 기반 확률과 검출 사유 리스트
        """
        if not ocr_texts:
            return 0.5, []  # 중립

        ocr_combined = " ".join(ocr_texts)
        compact_ocr = _compact_for_match(ocr_combined)
        negative_phrases = (
            '광고아님', '광고아닌', '광고없음', '광고안받',
            '협찬아님', '협찬아닌', '협찬없음', '협찬안받',
        )
        matched_probs = []
        matched_keywords = []
        reasons = []

        for keyword, prob, reason in _OCR_KEYWORDS:
            compact_keyword = _compact_for_match(keyword)
            if compact_keyword in ('광고', '협찬') and any(phrase in compact_ocr for phrase in negative_phrases):
                continue

            if compact_keyword and compact_keyword in compact_ocr:
                if any(
                    compact_keyword in matched_keyword or matched_keyword in compact_keyword
                    for matched_keyword in matched_keywords
                ):
                    continue

                matched_keywords.append(compact_keyword)
                matched_probs.append(prob)
                reasons.append(f"{reason} (p={prob:.0%})")

        if not matched_probs:
            return 0.5, []

        combined_logit = sum(_safe_logit(p) for p in matched_probs)
        p_ocr = _sigmoid(combined_logit)

        return p_ocr, reasons

    def url_to_probability(self, image_urls: list[str]) -> tuple[float, list[str]]:
        """본문 이미지 URL에서 체험단/리뷰 플랫폼 도메인을 탐지해 광고 확률을 산출합니다.

        OCR(이미지 다운로드+인식)과 달리 URL 문자열 매칭만으로 동작하므로 비용이
        거의 없고, 플랫폼 배너가 OCR로 안 읽히거나 선택에서 누락돼도 잡아냅니다.

        Args:
            image_urls: 본문에서 추출한 이미지 URL 리스트

        Returns:
            (p_url, reasons): URL 기반 확률과 검출 사유 리스트
        """
        if not image_urls:
            return 0.5, []

        joined = " ".join(u for u in image_urls if u).lower()
        if not joined:
            return 0.5, []

        matched_probs = []
        matched_markers = []
        reasons = []
        for marker, prob, reason in _PLATFORM_URL_RULES:
            m = marker.lower()
            if m in joined and m not in matched_markers:
                matched_markers.append(m)
                matched_probs.append(prob)
                reasons.append(f"{reason} (p={prob:.0%})")

        if not matched_probs:
            return 0.5, []

        # 한 URL이 여러 마커(예: reviewnote.cloud + /gongjeong)에 걸려도
        # '체험단 광고'라는 단일 결론이므로, log-odds 합산 대신 최댓값을 사용해
        # OCR '협찬' 검출과 동일한 강한 증거 등급(85%)을 유지한다.
        p_url = max(matched_probs)

        return p_url, reasons

    def _combine_logodds(self, p_model: float, p_rule: float, p_ocr: float, p_url: float) -> float:
        """네 소스의 확률을 Log-odds 가중합으로 결합한 '순수 결합 확률'.

        logit_final = w_model·logit(p_model) + w_rule·logit(p_rule)
                    + w_ocr·logit(p_ocr) + w_url·logit(p_url)

        중립(0.5)인 소스는 logit이 0이라 결과에 영향을 주지 않습니다.
        """
        logit_final = (
            self.w_model * _safe_logit(p_model)
            + self.w_rule * _safe_logit(p_rule)
            + self.w_ocr * _safe_logit(p_ocr)
            + self.w_url * _safe_logit(p_url)
        )
        return _sigmoid(logit_final)

    def fuse(self, p_model: float, p_rule: float, p_ocr: float, p_url: float = 0.5) -> float:
        """소스 확률을 Log-odds 가중 결합 후, 강한 법적 증거에 한해 임계치 초과를 보장합니다.

        순수 log-odds 결합만으로는 모델이 낮으면(예: 29%) 강한 증거(85%)와 섞여
        중간값(~49%)이 되어 결정 임계치(0.5)를 못 넘길 수 있습니다. 원고료/협찬/
        공정위/체험단 플랫폼처럼 '법적으로 광고가 확정되는' 증거(p >= 0.85)가 있으면
        최소 self.min_ad_prob(기본 0.55)는 보장합니다.

        단, 이전처럼 85%로 '고정'하지 않습니다. 결합값이 min_ad_prob보다 높으면
        그 결합값(모델 + 증거)을 그대로 사용하므로, 모델의 판단이 점수에 계속 반영됩니다.

        Args:
            p_model: 딥러닝 모델의 광고 확률 (sigmoid 출력)
            p_rule:  텍스트 규칙 기반 확률
            p_ocr:   OCR 기반 확률
            p_url:   이미지 URL(체험단 플랫폼 도메인) 기반 확률

        Returns:
            p_final: 결합값. 단, 강한 증거가 있으면 최소 min_ad_prob 보장.
        """
        p_final = self._combine_logodds(p_model, p_rule, p_ocr, p_url)

        strong_threshold = 0.85
        has_strong = any(p >= strong_threshold for p in (p_rule, p_ocr, p_url))
        if has_strong:
            p_final = max(p_final, self.min_ad_prob)

        return p_final

    @staticmethod
    def _humanize_reason(reason: str) -> str:
        """근거 문구에서 기술적 표기(p값·OCR/URL 접두사·'검출')를 떼어 읽기 쉽게 만든다.
        예: "이미지 OCR: '협찬' 검출 (p=85%)" → "이미지에서 '협찬'"
        """
        r = re.sub(r'\s*\(p=\d+%\)', '', reason or '')
        r = r.replace('이미지 OCR:', '이미지에서').replace('OCR:', '이미지에서')
        r = r.replace('URL:', '').replace(' 검출', '')
        return r.strip()

    def analyze(self, p_model: float, raw_content: str, ocr_texts: list[str],
                image_urls: list[str] = None) -> dict:
        """전체 분석을 수행하고 상세 리포트를 반환합니다.

        Args:
            p_model:     딥러닝 모델의 광고 확률
            raw_content: 원본 본문 텍스트
            ocr_texts:   OCR 추출 텍스트 리스트
            image_urls:  본문 이미지 URL 리스트 (체험단 플랫폼 도메인 탐지용)

        Returns:
            dict: {
                'p_model': float,
                'p_rule': float,
                'p_ocr': float,
                'p_url': float,
                'p_combined': float,   # 하한선 적용 전 순수 결합 확률
                'p_final': float,      # 하한선 적용 후 최종 확률
                'viral_score': int,
                'reasons': list[str],
                'summary': str
            }
        """
        p_rule, rule_reasons = self.keyword_to_probability(raw_content)
        p_ocr, ocr_reasons = self.ocr_to_probability(ocr_texts)
        p_url, url_reasons = self.url_to_probability(image_urls or [])

        p_combined = self._combine_logodds(p_model, p_rule, p_ocr, p_url)
        p_final = self.fuse(p_model, p_rule, p_ocr, p_url)

        all_reasons = rule_reasons + ocr_reasons + url_reasons
        viral_score = int(round(p_final * 100))
        viral_score = max(0, min(100, viral_score))

        # 사람이 읽기 쉬운 요약 생성: 점수대별 헤드라인 + 광고 신호 목록
        if viral_score >= 71:
            headline = f"광고·협찬 가능성이 높아요 (종합 {viral_score}점)"
        elif viral_score >= 31:
            headline = f"광고·협찬이 의심돼요 (종합 {viral_score}점)"
        else:
            headline = f"광고 신호가 거의 없어요 (종합 {viral_score}점)"

        # 광고 신호 발견 여부와 무관하게 모델 단독 예측을 항상 표시
        model_line = f"AI 모델 단독 예측 {round(p_model * 100)}%"

        # 근거 문구를 정리하고 중복 제거(순서 보존)
        bullets = []
        seen = set()
        for r in all_reasons:
            clean = self._humanize_reason(r)
            if clean and clean not in seen:
                seen.add(clean)
                bullets.append(clean)

        if bullets:
            body = "발견된 광고 신호\n" + "\n".join(f"• {b}" for b in bullets)
        else:
            body = "본문 외 광고 문구·배너는 발견되지 않았습니다."

        summary = f"{headline}\n{model_line}\n{body}"

        return {
            'p_model': p_model,
            'p_rule': p_rule,
            'p_ocr': p_ocr,
            'p_url': p_url,
            'p_combined': p_combined,
            'p_final': p_final,
            'viral_score': viral_score,
            'reasons': all_reasons,
            'summary': summary,
        }


# =========================================================
# 모듈 단독 실행 시 간단한 테스트
# =========================================================
if __name__ == "__main__":
    fusion = ProbabilityFusion()

    print("=" * 60)
    print("🧪 소프트 확률 결합 테스트")
    print("=" * 60)

    # 테스트 1: 모델 확률만 (규칙/OCR 증거 없음)
    result = fusion.analyze(0.65, "맛있는 치킨 리뷰입니다", [])
    print(f"\n[테스트 1] 모델만: {result['summary']}")
    print(f"  → 최종 점수: {result['viral_score']}점")

    # 테스트 2: 모델 + 강한 텍스트 증거
    result = fusion.analyze(0.30, "본 포스팅은 소정의 원고료를 제공받아 작성되었습니다", [])
    print(f"\n[테스트 2] 모델 약 + 텍스트 강: {result['summary']}")
    print(f"  → 최종 점수: {result['viral_score']}점")

    # 테스트 3: 모델 + OCR 증거
    result = fusion.analyze(0.45, "맛있는 음식 리뷰", ["원고료를 지원받아 작성"])
    print(f"\n[테스트 3] 모델 중립 + OCR: {result['summary']}")
    print(f"  → 최종 점수: {result['viral_score']}점")

    # 테스트 4: 내돈내산 (음의 증거)
    result = fusion.analyze(0.40, "내돈내산으로 구매한 제품입니다", [])
    print(f"\n[테스트 4] 내돈내산: {result['summary']}")
    print(f"  → 최종 점수: {result['viral_score']}점")

    # 테스트 5: 패널티 방식과 비교
    print("\n" + "=" * 60)
    print("📊 기존 패널티 방식 vs 소프트 확률 결합 비교")
    print("=" * 60)

    test_cases = [
        ("모델 30% + 원고료 검출", 0.30, "소정의 원고료를 제공받아 작성", []),
        ("모델 25% + OCR 협찬", 0.25, "맛집 리뷰", ["협찬받아 작성"]),
        ("모델 70% + 증거 없음", 0.70, "맛집 리뷰", []),
        ("모델 60% + 내돈내산", 0.60, "내돈내산 리뷰입니다", []),
    ]

    print(f"\n{'시나리오':<30} {'기존(패널티)':<15} {'신규(결합)':<15}")
    print("-" * 60)
    for name, p_model, content, ocr in test_cases:
        result = fusion.analyze(p_model, content, ocr)

        # 기존 패널티 방식 시뮬레이션
        old_score = int(p_model * 100)
        flat = content.replace(" ", "")
        if re.search(r'소정의\s*원고료', content): old_score += 25
        if re.search(r'제공받아\s*작성', content): old_score += 25
        if '협찬' in " ".join(ocr): old_score += 20
        if '내돈내산' in flat: old_score -= 10
        old_score = max(0, min(100, old_score))

        print(f"{name:<30} {old_score:>6}점       {result['viral_score']:>6}점")
