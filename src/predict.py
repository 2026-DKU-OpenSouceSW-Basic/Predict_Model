# -*- coding: utf-8 -*-
import os
import re
import tempfile
import time
import torch
import requests
import numpy as np
import shap
from transformers import AutoTokenizer
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

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
preprocessor = BlogPreprocessor()

print("[INFO] OCR 모델 및 크롬 브라우저를 로드합니다...")
ocr = PaddleOCR(lang="korean", use_textline_orientation=True)

chrome_options = Options()
chrome_options.add_argument("--headless=new") 
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# =========================================================
# 2. 크롤링 함수
# =========================================================
def extract_text_from_image(img_url):
    if not img_url or not img_url.startswith("http"): return ""
    try:
        img_headers = {"Referer": "https://blog.naver.com/"}
        img_resp = requests.get(img_url, headers=img_headers, timeout=10)
        if img_resp.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(img_resp.content)
                tmp_path = tmp_file.name
            result = ocr.ocr(tmp_path)
            os.remove(tmp_path)
            if result and result[0]:
                extracted_texts = [line[1][0] for line in result[0]]
                return " ".join(extracted_texts)
    except Exception:
        pass
    return ""

def scrape_single_post(blog_url):
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
        
        js_script = """
        let container = arguments[0];
        let imgs = Array.from(container.getElementsByTagName('img'));
        for(let i = 0; i < imgs.length; i++) {
            let src = imgs[i].getAttribute('data-lazy-src') || imgs[i].src || '';
            let textNode = document.createTextNode('\\n[NAVER_IMG_URL:' + src + ']\\n');
            imgs[i].parentNode.replaceChild(textNode, imgs[i]);
        }
        """
        driver.execute_script(js_script, content_element)
        content = content_element.text
        
        matches = re.findall(r"\[NAVER_IMG_URL:(.*?)\]", content)
        if matches:
            total_imgs = len(matches)
            target_indices = set(range(total_imgs)) if total_imgs <= 3 else {0, 1, total_imgs - 1}
            for i, img_url in enumerate(matches):
                replace_str = f"\n[image_{i + 1}]\n"
                if i in target_indices and img_url:
                    ocr_text = extract_text_from_image(img_url)
                    if ocr_text.strip(): replace_str = f"\n[이미지 텍스트: {ocr_text}]\n"
                content = content.replace(f"[NAVER_IMG_URL:{img_url}]", replace_str, 1)

        return title.strip(), content.strip()
    except Exception as e:
        return "추출 실패", f"{e}"

# =========================================================
# 3. SHAP 연동을 위한 파이토치 모델 래퍼 (Wrapper)
# =========================================================
# SHAP 연산 중 변동을 막기 위한 고정 전역 변수 설정
_fixed_global_features = None
_fixed_title = ""  # [수정] 원본 제목을 고정 보관할 변수 추가

def predict_wrapper(texts):
    """SHAP이 텍스트를 마스킹해가며 수백 번 호출할 함수입니다."""
    global _fixed_global_features
    global _fixed_title  # 전역 변수 참조
    probs = []
    for text in texts:
        # [수정] 기존 "" 대신 고정된 원본 제목(_fixed_title)을 전달하여 본문 내 [TARGET] 치환 활성화
        processed = preprocessor.process(_fixed_title, text)
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

        # 글로벌 피처 고정 로직
        if _fixed_global_features is not None:
            global_features = _fixed_global_features.to(device)
        else:
            gf = processed['global_features']
            global_features = torch.tensor([[
                gf['total_images'] / 100.0, gf['total_length'] / 8000.0, gf['paragraph_count'] / 50.0, 
                gf['target_frequency'] / 40.0, gf['sentiment_score']
            ]], dtype=torch.float).to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, local_features, global_features)
            prob = torch.sigmoid(logits).item()
        
        probs.append(prob)
        
    return np.array(probs)

# SHAP Explainer 전역 객체 생성 (server.py 등에서 사용 가능)
masker = shap.maskers.Text(r"\s+")
explainer = shap.Explainer(predict_wrapper, masker)

# =========================================================
# 4. SHAP 분석 및 리포트 생성
# =========================================================
def check_blog_ad_with_shap(title, body):
    global _fixed_global_features
    global _fixed_title  # 전역 변수 참조
    
    full_text = f"제목: {title}\n본문: {body}"
    
    # [수정] 원본 제목을 미리 고정하여 래퍼 함수에서 [TARGET] 추출에 사용하도록 설정
    _fixed_title = title
    
    original_processed = preprocessor.process(title, body)
    gf = original_processed['global_features']
    _fixed_global_features = torch.tensor([[
        gf['total_images'] / 100.0, gf['total_length'] / 8000.0, gf['paragraph_count'] / 50.0,
        gf['target_frequency'] / 40.0, gf['sentiment_score']
    ]], dtype=torch.float)
    print(f"📊 [고정 피처] 이미지={gf['total_images']}, 길이={gf['total_length']}, "
          f"문단={gf['paragraph_count']}, 타겟빈도={gf['target_frequency']}, 감성={gf['sentiment_score']}")
    
    # 원본 확률 연산
    original_prob = predict_wrapper([full_text])[0]
    
    print("=" * 60)
    print(f"📌 분석된 제목: {title}")
    if original_prob > 0.5:
        print(f"🚨 AI 판별 결과: [광고 / 협찬]일 확률이 {original_prob*100:.1f}% 입니다.")
    else:
        print(f"✅ AI 판별 결과: [내돈내산 / 순수리뷰]일 확률이 {(1-original_prob)*100:.1f}% 입니다.")
    print("=" * 60)

    print("\n🧠 [SHAP 분석 중] 단어들의 상호작용과 진짜 기여도를 계산하고 있습니다...")
    print("   (텍스트 길이에 따라 약 30초 ~ 2분 정도 소요될 수 있습니다. 잠시만 기다려주세요!)")
    
    # 전역 explainer를 사용하여 SHAP 분석 진행
    shap_values = explainer([full_text])
    
    # [수정] 분석 완료 후 전역 고정 상태 해제
    _fixed_global_features = None
    _fixed_title = ""
    
    # 1. 터미널 출력용 요약 데이터 정리
    tokens = shap_values.data[0]
    values = shap_values.values[0]
    
    word_impacts = list(zip(tokens, values))
    word_impacts = [x for x in word_impacts if x[0].strip() and not x[0].startswith("[")]
    word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("-" * 60)
    print("🔎 [SHAP 게임이론 XAI] 광고/내돈내산 판별에 가장 크게 기여한 핵심 단어 TOP 10")
    for token, impact in word_impacts[:10]:
        impact_percent = abs(impact) * 100
        if impact > 0:
            print(f" 📈 '{token}': [광고] 성향 ➕ {impact_percent:.1f}%p 기여")
        else:
            print(f" 📉 '{token}': [내돈내산] 성향 ➖ {impact_percent:.1f}%p 기여")
    print("-" * 60)
    
    # 2. 브라우저용 시각화 리포트 저장
    html_content = shap.plots.text(shap_values, display=False)
    report_filename = "shap_report.html"
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n🎉 분석 완료! 아름다운 시각화 리포트가 생성되었습니다.")
    print(f"👉 프로젝트 폴더 내의 '{report_filename}' 파일을 웹 브라우저(크롬 등)로 열어보세요!")

# =========================================================
# 5. 실전 실행 프로세스
# =========================================================
if __name__ == "__main__":
    target_url = input("🔗 판별할 네이버 블로그 링크(URL)를 입력하세요: ")
    
    print(f"\n[데이터 수집] 웹페이지에서 실시간 본문 및 OCR 데이터를 긁어옵니다...")
    scraped_title, scraped_content = scrape_single_post(target_url)
    
    if "추출 실패" in scraped_title:
        print(f"❌ 크롤링 실패: {scraped_content}")
    else:
        check_blog_ad_with_shap(scraped_title, scraped_content)
        
    driver.quit()