# -*- coding: utf-8 -*-
import os

# PaddlePaddle 3.x의 새 실행엔진(PIR)이 oneDNN(MKLDNN) 가속 경로와 충돌해
# 'ConvertPirAttribute2RuntimeAttribute not support' 에러로 OCR이 전부 실패하는 문제 방지.
# 반드시 paddleocr/paddle import 이전에 설정해야 적용된다.
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import re
import time
import tempfile
import torch
import requests
import queue
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from paddleocr import PaddleOCR

# 커스텀 모듈 임포트
from model import BlogAdClassifier
from preprocessing import BlogPreprocessor
from dataset import normalize_global_features, build_tokenizer
from probability_fusion import ProbabilityFusion

# =========================================================
# 1. 환경 설정 및 초기화
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("[INFO] AI 분류 모델 및 토크나이저를 로드합니다...")
model = BlogAdClassifier(global_feature_dim=5).to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "blog_ad_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tokenizer = build_tokenizer()
preprocessor = BlogPreprocessor()
fusion = ProbabilityFusion(w_model=1.0, w_rule=0.6, w_ocr=0.5)

print("[INFO] OCR 모델 및 크롬 브라우저를 로드합니다...")

def create_ocr_reader():
    # enable_mkldnn=False로 oneDNN 가속을 끄면 PIR-oneDNN 충돌 에러가 사라진다.
    # PaddleOCR 버전마다 지원 파라미터가 달라 단계적으로 fallback한다.
    try:
        return PaddleOCR(lang="korean", use_textline_orientation=True,
                         device="cpu", enable_mkldnn=False)
    except TypeError:
        pass
    try:
        return PaddleOCR(lang="korean", use_angle_cls=True,
                         use_gpu=False, enable_mkldnn=False)
    except TypeError:
        pass
    try:
        return PaddleOCR(lang="korean", use_textline_orientation=True, device="cpu")
    except TypeError:
        return PaddleOCR(lang="korean", use_angle_cls=True, use_gpu=False)

ocr = create_ocr_reader()

OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.45"))
OCR_HEAD_IMAGES = int(os.environ.get("OCR_HEAD_IMAGES", "2"))
OCR_TAIL_IMAGES = int(os.environ.get("OCR_TAIL_IMAGES", "2"))

# OCR 대상에서 제외할 정크 이미지(프로필/스티커/지도/정적 에셋) URL 마커
_JUNK_IMAGE_MARKERS = (
    "blogpfthumb",              # 블로거 프로필 썸네일
    "storep-phinf",             # OGQ 스티커
    "static.map", "staticmap",  # 네이버 지도 캡처
    "ssl.pstatic.net/static",   # 네이버 정적 에셋/아이콘
)

def _is_content_image(url):
    """프로필/스티커/지도/빈 URL을 제외한 '본문 콘텐츠 이미지'인지 판별."""
    if not url or not url.startswith("http"):
        return False
    return not any(marker in url for marker in _JUNK_IMAGE_MARKERS)

# OGQ 스티커 도메인. 대부분 장식용이지만 협찬 고지("본 포스팅은 소정의 원고료를
# 지원받아...")를 OGQ 스티커 이미지로 붙이는 블로그가 많다. 썸네일 후보에선 계속
# 제외(_is_content_image=False)하되, OCR은 반드시 수행해야 협찬을 놓치지 않는다.
_OGQ_STICKER_MARKER = "storep-phinf"

def _is_ogq_sticker(url):
    """OGQ 스티커 이미지 URL인지 판별(OCR 대상 포함용)."""
    return bool(url) and url.startswith("http") and _OGQ_STICKER_MARKER in url

# 네이버 블로그에서 '본문 업로드 사진'으로 신뢰할 수 있는 도메인(화이트리스트).
# 프로필/스티커/지도/아이콘은 모두 다른 호스트를 쓰므로, 이 도메인이면 사진이 거의 확실하다.
_PHOTO_HOST_MARKERS = (
    "postfiles.pstatic.net",        # PC 본문 업로드 사진
    "blogfiles.pstatic.net",        # 본문 첨부 사진
    "mblogthumb-phinf.pstatic.net", # 모바일 본문 사진(우리는 m.blog로 스크랩)
)

def _is_photo_image(url):
    """본문 '사진' 도메인 화이트리스트에 드는지 판별(썸네일 선택 전용)."""
    if not _is_content_image(url):
        return False
    return any(host in url for host in _PHOTO_HOST_MARKERS)

def _select_thumbnail(urls):
    """프론트 전송용 대표 썸네일 1장을 선택한다.

    1순위: 본문 사진 도메인(화이트리스트)만 추려 그 안에서 3번째(없으면 첫 장).
           - 첫 1~2장은 인사말/배너인 경우가 많아 기존처럼 3번째를 우선한다.
    2순위(fallback): 화이트리스트에 하나도 안 걸리면, 정크만 제외한 목록에서 동일 규칙.
                     도메인이 새로 생겨도 최소한 기존 동작은 보장한다.
    """
    for candidates in (
        [u for u in urls if _is_photo_image(u)],     # 화이트리스트(사진) 우선
        [u for u in urls if _is_content_image(u)],   # negative 필터 fallback
    ):
        if len(candidates) >= 3:
            return candidates[2]
        if candidates:
            return candidates[0]
    return ""

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-software-rasterizer")
chrome_options.add_argument("--mute-audio")
chrome_options.add_argument("--window-size=1920,1080")

DRIVER_POOL_SIZE = 1
driver_pool = queue.Queue(maxsize=DRIVER_POOL_SIZE)

for _ in range(DRIVER_POOL_SIZE):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver_pool.put(driver)

def clean_ocr_text(text):
    text = re.sub(r'\s+', ' ', text or '')
    text = re.sub(r'\s+([,.!?%])', r'\1', text)
    return text.strip()

def _extract_paddle_ocr_lines(result):
    lines = []

    def add_line(text, score=None):
        text = clean_ocr_text(str(text))
        try:
            score = 1.0 if score is None else float(score)
        except (TypeError, ValueError):
            score = 1.0

        if text and score >= OCR_MIN_CONFIDENCE:
            lines.append((text, score))

    def walk(node):
        if hasattr(node, "to_dict"):
            walk(node.to_dict())
            return

        if hasattr(node, "json") and not isinstance(node, (str, bytes)):
            json_data = node.json() if callable(node.json) else node.json
            walk(json_data)
            return

        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts")
            scores = node.get("rec_scores") or node.get("scores") or []
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            if isinstance(texts, (list, tuple)):
                for idx, text in enumerate(texts):
                    score = scores[idx] if idx < len(scores) else None
                    add_line(text, score)
                return

            for value in node.values():
                walk(value)
            return

        if isinstance(node, (list, tuple)):
            if len(node) >= 2:
                # PaddleOCR 2.x: [box, (text, score)]
                if isinstance(node[1], (list, tuple)) and len(node[1]) >= 2 and isinstance(node[1][0], str):
                    add_line(node[1][0], node[1][1])
                    return

                # Some wrappers return (text, score).
                if isinstance(node[0], str):
                    add_line(node[0], node[1])
                    return

            for child in node:
                walk(child)

    walk(result)
    return lines

def _select_ocr_indices(matches):
    """OCR 대상 이미지의 원본 인덱스 집합을 반환한다.

    (1) 콘텐츠 사진: 앞 OCR_HEAD_IMAGES장 + 뒤 OCR_TAIL_IMAGES장.
        체험단 배너는 본문 맨 위/맨 아래에 오는 경우가 대부분이라 앞뒤만 선별해
        OCR 비용과 오탐을 함께 줄인다.
    (2) OGQ 스티커: 위치와 무관하게 전부. 장식용이 많지만 협찬 고지를 OGQ
        스티커로 붙이는 경우가 많아, 누락하면 광고를 통째로 놓친다(수가 적어 비용도 작음).
    """
    selected = set()

    content_indices = [i for i, url in enumerate(matches) if _is_content_image(url)]
    if content_indices:
        head = max(0, OCR_HEAD_IMAGES)
        tail = max(0, OCR_TAIL_IMAGES)
        if len(content_indices) <= head + tail:
            selected.update(content_indices)
        else:
            selected.update(content_indices[:head] + content_indices[-tail:])

    # OGQ 스티커는 전부 OCR (협찬 고지 배너 누락 방지)
    selected.update(i for i, url in enumerate(matches) if _is_ogq_sticker(url))

    return selected

def extract_text_from_image(img_url):
    if not img_url or not img_url.startswith("http"):
        return ""

    tmp_path = None
    try:
        img_headers = {
            "Referer": "https://blog.naver.com/",
            "User-Agent": "Mozilla/5.0"
        }
        img_resp = requests.get(img_url, headers=img_headers, timeout=10)
        img_resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(img_resp.content)
            tmp_path = tmp_file.name

        result = ocr.ocr(tmp_path)
        extracted_lines = _extract_paddle_ocr_lines(result)

        unique_texts = []
        seen = set()
        for text, _score in extracted_lines:
            key = re.sub(r'\s+', '', text).lower()
            if key and key not in seen:
                seen.add(key)
                unique_texts.append(text)

        return clean_ocr_text(" ".join(unique_texts))
    except Exception as e:
        print(f"[OCR WARNING] 이미지 OCR 실패: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return ""

def scrape_single_post(blog_url):

    driver = driver_pool.get()
    try:
        blog_url = blog_url.strip()
        if "blog.naver.com" in blog_url and "m.blog.naver.com" not in blog_url:
            blog_url = blog_url.replace("blog.naver.com", "m.blog.naver.com")

        driver.get(blog_url)
        title_element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h2.se-title-text, h2.title, div.se-title-text, span.se-combined-title-text"))
        )
        title = title_element.get_attribute("title") or title_element.text

        content_element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.se-main-container, div.post_ct"))
        )

        # lazy-load 안정화: 네이버는 스크롤해야 이미지가 DOM에 채워진다.
        # 컨테이너 내 <img> 개수가 더 늘지 않을 때까지 끝까지 스크롤해
        # total_images 등 글로벌 피처가 매 실행마다 흔들리는 문제를 막는다.
        # 이미지 개수가 '연속 2회' 동일할 때까지 스크롤한다. 1회 정체로 끊으면
        # OGQ 스티커 등 하단 콘텐츠가 늦게 로드될 때 누락된다(협찬 배너 누락의 원인).
        prev_count = -1
        stable = 0
        for _ in range(20):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            cur_count = driver.execute_script(
                "return arguments[0].getElementsByTagName('img').length;", content_element)
            if cur_count == prev_count:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev_count = cur_count

        js_script = """
        let container = arguments[0];
        let imgs = Array.from(container.getElementsByTagName('img'));
        for(let i = 0; i < imgs.length; i++) {
            let src = imgs[i].getAttribute('data-lazy-src') || imgs[i].getAttribute('data-src') || imgs[i].src || '';
            let textNode = document.createTextNode('\\n[NAVER_IMG_URL:' + src + ']\\n');
            imgs[i].parentNode.replaceChild(textNode, imgs[i]);
        }
        """
        driver.execute_script(js_script, content_element)
        content = content_element.text

        ocr_texts = []
        matches = re.findall(r"\[NAVER_IMG_URL:(.*?)\]", content)
        image_urls = [u for u in matches if u and u.startswith("http")]

        # 프론트 전송용 대표 썸네일 선택(본문 사진 도메인 우선 → 정크 제외 fallback)
        thumbnail_url = _select_thumbnail(matches)
        if matches:
            target_indices = _select_ocr_indices(matches)
            for i, img_url in enumerate(matches):
                replace_str = f"\n[image_{i + 1}]\n"
                if img_url and i in target_indices:
                    ocr_text = extract_text_from_image(img_url)
                    if ocr_text:
                        ocr_texts.append(ocr_text)

                content = content.replace(f"[NAVER_IMG_URL:{img_url}]", replace_str, 1)

        return title.strip(), content.strip(), ocr_texts, image_urls, thumbnail_url
    except Exception as e:
        return "추출 실패", f"{e}", [], [], ""
    finally:
        driver_pool.put(driver)

def predict_ad(title, body):
    processed = preprocessor.process(title, body)
    paragraphs = processed['paragraphs']

    if len(paragraphs) > 10:
        paragraphs = paragraphs[:5] + paragraphs[-5:]

    input_ids_list, attention_mask_list, local_features_list = [], [], []

    for para in paragraphs:
        encoded = tokenizer(para['text'], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        input_ids_list.append(encoded['input_ids'].squeeze(0))
        attention_mask_list.append(encoded['attention_mask'].squeeze(0))
        local_features_list.append([para.get('image_count', 0)])

    while len(input_ids_list) < 10:
        input_ids_list.append(torch.zeros(128, dtype=torch.long))
        attention_mask_list.append(torch.zeros(128, dtype=torch.long))
        local_features_list.append([0])

    input_ids = torch.stack(input_ids_list).unsqueeze(0).to(device)
    attention_mask = torch.stack(attention_mask_list).unsqueeze(0).to(device)
    local_features = torch.tensor(local_features_list, dtype=torch.float).unsqueeze(0).to(device)

    gf = processed['global_features']
    norm_gf = normalize_global_features(gf)
    global_features = torch.tensor([norm_gf], dtype=torch.float).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, local_features, global_features)
        prob = torch.sigmoid(logits).item()

    return prob, gf

# =========================================================
# 4. AI 광고 판별 및 리포팅
# =========================================================
def check_blog_ad(title, body, ocr_texts=None, image_urls=None):
    ocr_texts = ocr_texts or []
    image_urls = image_urls or []
    prob, gf = predict_ad(title, body)
    fusion_result = fusion.analyze(prob, body, ocr_texts, image_urls)
    final_prob = fusion_result['p_final']

    print("=" * 60)
    print(f"📌 분석된 제목: {title}")
    if final_prob > 0.5:
        print(f"🚨 AI 판별 결과: [광고 / 협찬]일 확률이 {final_prob*100:.1f}% 입니다.")
    else:
        print(f"✅ AI 판별 결과: [내돈내산 / 순수리뷰]일 확률이 {(1-final_prob)*100:.1f}% 입니다.")
    print(f"🧠 모델 단독 확률: {prob*100:.1f}%")
    if ocr_texts:
        print(f"🔎 OCR 추출 텍스트: {' | '.join(ocr_texts[:5])}")
    print(f"🧾 판단 근거: {fusion_result['summary']}")
    print("=" * 60)
    print(f"📊 [글로벌 피처] 이미지={gf['total_images']}, 길이={gf['total_length']}, "
          f"문단={gf['paragraph_count']}, 타겟빈도={gf['target_frequency']}, 감성={gf['sentiment_score']}")
    print("=" * 60)

def close_driver_pool():
    while not driver_pool.empty():
        driver = driver_pool.get_nowait()
        driver.quit()

# =========================================================
# 5. 실전 실행 프로세스
# =========================================================
if __name__ == "__main__":
    target_url = input("🔗 판별할 네이버 블로그 링크(URL)를 입력하세요: ")

    print(f"\n[데이터 수집] 웹페이지에서 실시간 본문 및 OCR 데이터를 긁어옵니다...")
    scraped_title, scraped_content, ocr_texts, image_urls, thumbnail_url = scrape_single_post(target_url)

    if "추출 실패" in scraped_title:
        print(f"❌ 크롤링 실패: {scraped_content}")
    else:
        check_blog_ad(scraped_title, scraped_content, ocr_texts, image_urls)

    close_driver_pool()
