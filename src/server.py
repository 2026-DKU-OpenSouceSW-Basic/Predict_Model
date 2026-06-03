# -*- coding: utf-8 -*-
# C:\Users\Yun\OpenSourceSW\src\server.py
import os
import urllib.parse
import hashlib
import requests
import threading
import uvicorn
import shap
import torch
import asyncio
import json
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

# predict.py의 전역 인스턴스 및 함수 임포트
import predict
from predict import (
    scrape_single_post,
    predict_wrapper,
    preprocessor
)

app = FastAPI(title="Viral Product Filtering AI Analyzer")

# 동시 SHAP 연산 시 predict.py의 전역 변수(고정 피처 등)가 충돌하지 않도록 방지하는 락(Lock)
shap_lock = threading.Lock()

# 네이버 블로그 검색 API 키
NAVER_CLIENT_ID = "qdrPVw0WtyIj5d3l1uoM"
NAVER_CLIENT_SECRET = "czfuqupEJa"

class AnalysisItem(BaseModel):
    title: str
    link: str
    body: str
    bloggerName: str
    postDate: str
    viralScore: int
    reason: str

# def run_shap_background(title: str, body: str, file_id: str):
#     """백그라운드에서 실행되는 무거운 SHAP 분석 태스크"""
#     print(f"[SHAP] 백그라운드 SHAP 분석 시작 (ID: {file_id})")
#     
#     with shap_lock:
#         try:
#             # 1. 고정 피처 설정 (predict.py 내부 변수 갱신)
#             original_processed = preprocessor.process(title, body)
#             gf = original_processed['global_features']
#             
#             predict._fixed_title = title
#             predict._fixed_global_features = torch.tensor([[
#                 gf['total_images'] / 100.0, gf['total_length'] / 8000.0, gf['paragraph_count'] / 50.0,
#                 gf['target_frequency'] / 40.0, gf['sentiment_score']
#             ]], dtype=torch.float)
#             
#             full_text = f"제목: {title}\n본문: {body}"
#             
#             # 2. SHAP 분석 실행
#             shap_values = explainer([full_text])
#             
#             # 3. 분석 완료 후 predict.py 전역 변수 초기화
#             predict._fixed_global_features = None
#             predict._fixed_title = ""
#             
#             # 4. HTML 파일로 시각화 결과 저장
#             html_content = shap.plots.text(shap_values, display=False)
#             report_filename = f"shap_report_{file_id}.html"
#             
#             with open(report_filename, "w", encoding="utf-8") as f:
#                 f.write(html_content)
#                 
#             print(f"[SHAP] 분석 완료! 리포트 저장됨: {report_filename}")
#         except Exception as e:
#             print(f"[SHAP ERROR] 백그라운드 SHAP 분석 실패: {e}")
#             predict._fixed_global_features = None
#             predict._fixed_title = ""

async def event_generator(query: str, background_tasks: BackgroundTasks):
    """검색어 수집, 크롤링 및 AI 분석 결과를 1개씩 실시간으로 SSE 포맷으로 yield하는 제너레이터"""
    try:
        # Step 0: 검색 시작
        yield "event: progress\ndata: " + json.dumps({"step": 0, "message": "검색 시작"}) + "\n\n"
        await asyncio.sleep(0.1)
        
        # 네이버 API를 통해 검색어에 매칭되는 블로그 20개 탐색
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://openapi.naver.com/v1/search/blog?query={encoded_query}&display=20&start=1&sort=sim"
        
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        
        # 외부 API 호출이므로 비동기로 실행
        resp = await asyncio.to_thread(requests.get, search_url, headers=headers)
        
        if resp.status_code != 200:
            print(f"[API ERROR] 네이버 블로그 검색 API 호출 실패: {resp.status_code}")
            yield "event: progress\ndata: " + json.dumps({"step": 3, "message": f"검색 실패 (HTTP {resp.status_code})"}) + "\n\n"
            return
            
        items = resp.json().get("items", [])
        if not items:
            print("[API] 검색 결과 없음. Fallback Mock 데이터를 사용합니다.")
            items = [
                {"title": "내돈내산 진짜 감동받은 가성비 무선이어폰 추천 후기", "link": "https://blog.naver.com/pure_buyer/223019283921", "bloggername": "현명한소비자", "postdate": "2026-05-20"},
                {"title": "[제품협찬] 성능 좋은 무선이어폰 실사용 솔직 후기", "link": "https://blog.naver.com/sponsored_blogger/223019283923", "bloggername": "체험단매니아", "postdate": "2026-05-15"}
            ]
            
        # Step 1: 결과 수집 중
        yield "event: progress\ndata: " + json.dumps({"step": 1, "message": "결과 수집 중..."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        # Step 2: 데이터 분석 중
        yield "event: progress\ndata: " + json.dumps({"step": 2, "message": "데이터 분석 중..."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        # 1개씩 순서대로 크롤링 및 실시간 분석 전송
        for idx, item in enumerate(items, 1):
            link = item.get("link", "")
            blogger_name = item.get("bloggername", "")
            post_date = item.get("postdate", "")
            raw_title = item.get("title", "")
            
            print(f"[API] ({idx}/{len(items)}) 분석 시작: {link}")
            
            # 1. 크롤링 진행 (Selenium + OCR) - 메인 루프를 블로킹하지 않도록 서브스레드에서 실행
            scraped_title, scraped_content = await asyncio.to_thread(scrape_single_post, link)
            
            if "추출 실패" in scraped_title:
                # 크롤링 오류 등으로 파싱이 막힌 경우 API 검색에 담겨온 제목을 fallback으로 클리닝하여 사용
                clean_title = raw_title.replace("<b>", "").replace("</b>", "") if isinstance(raw_title, str) else "블로그 글"
                scraped_title = clean_title if clean_title.strip() else "블로그 글"
                scraped_content = item.get("description", "").replace("<b>", "").replace("</b>", "")
                if not scraped_content.strip():
                    print(f"[API] ({idx}/{len(items)}) 건너뜀 (크롤링 실패 및 본문 없음)")
                    continue
            
            # 2. 모델 예측값 계산 (비동기 스레드 풀에서 계산)
            original_processed = await asyncio.to_thread(preprocessor.process, scraped_title, scraped_content)
            gf = original_processed['global_features']
            
            def get_prediction():
                predict._fixed_title = scraped_title
                predict._fixed_global_features = torch.tensor([[
                    gf['total_images'] / 100.0, gf['total_length'] / 8000.0, gf['paragraph_count'] / 50.0,
                    gf['target_frequency'] / 40.0, gf['sentiment_score']
                ]], dtype=torch.float)
                prob = predict_wrapper([f"제목: {scraped_title}\n본문: {scraped_content}"])[0]
                predict._fixed_global_features = None
                predict._fixed_title = ""
                return prob
                
            prob = await asyncio.to_thread(get_prediction)
            viral_score = int(prob * 100)
            
            # 3. 고유 파일 ID 및 리포트명 생성
            link_hash = hashlib.md5(link.encode('utf-8')).hexdigest()[:10]
            report_file = f"shap_report_{link_hash}.html"
            
            # 4. 빠른 피드백을 위한 키워드 검출 분석
            keywords = ["소정의 원고료", "경제적 대가", "무상으로", "제품을 협찬", "체험단", "내돈내산", "광고"]
            detected = [kw for kw in keywords if kw.replace(" ", "") in scraped_content.replace(" ", "")]
            
            if detected:
                quick_reason = f"검출 키워드: {', '.join(detected)}"
            else:
                quick_reason = "특이 키워드 검출 없음"
                
            # 5. SHAP 분석 비활성화됨
            # background_tasks.add_task(run_shap_background, scraped_title, scraped_content, link_hash)
            
            # 6. 카드 데이터 전송 양식 세팅
            card_data = {
                "title": scraped_title,
                "link": link,
                "body": scraped_content[:150] + "...",
                "bloggerName": blogger_name,
                "postDate": post_date,
                "viralScore": viral_score,
                "reason": quick_reason
            }
            
            print(f"[API] ({idx}/{len(items)}) 분석 완료 및 전송: {scraped_title} (점수: {viral_score}점)")
            # SSE 형식에 맞게 yield
            yield f"event: card\ndata: {json.dumps(card_data)}\n\n"
            await asyncio.sleep(0.1)
            
        # Step 3: 모든 작업 완료
        yield "event: progress\ndata: " + json.dumps({"step": 3, "message": "결과 저장 중..."}) + "\n\n"
        
    except Exception as e:
        print(f"[API ERROR] 스트림 처리 중 에러 발생: {e}")
        yield "event: progress\ndata: " + json.dumps({"step": 3, "message": f"오류 발생: {e}"}) + "\n\n"

@app.get("/api/v1/analyze/stream")
def analyze_stream(background_tasks: BackgroundTasks, query: str = Query(..., description="검색어")):
    print(f"[API] 실시간 SSE 스트리밍 요청 수신: {query}")
    return StreamingResponse(
        event_generator(query, background_tasks),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
