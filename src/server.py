# -*- coding: utf-8 -*-
# C:\Users\Yun\OpenSourceSW\src\server.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import urllib.parse
import requests
import uvicorn
import asyncio
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

# predict.py의 전역 인스턴스 및 함수 임포트
from predict import (
    scrape_single_post,
    predict_ad
)
from probability_fusion import ProbabilityFusion

# 소프트 확률 결합 모듈 초기화
fusion = ProbabilityFusion(w_model=1.0, w_rule=0.6, w_ocr=0.5)

app = FastAPI(title="Viral Product Filtering AI Analyzer")

# 검색 API 연동 및 서비스 레이어
# 네이버 블로그 검색 API 키
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "qdrPVw0WtyIj5d3l1uoM")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "czfuqupEJa")

async def _await_with_keepalive(blocking_fn, *args, interval=10):
    """블로킹 작업(크롤링/추론)을 스레드에서 실행하는 동안, 끝날 때까지
    interval초마다 SSE keepalive 코멘트(': ...')를 yield한다.

    포스트 1건 분석이 수십 초씩 걸려 그동안 스트림이 '무음'이 되면,
    중계 백엔드의 read timeout이 발동해 한 장만 받고 끊긴다(로그의 timeout 원인).
    keepalive 바이트를 흘려 read timeout을 계속 리셋한다.
    완료 시 마지막에 ('result', 반환값)을 yield한다.
    """
    task = asyncio.ensure_future(asyncio.to_thread(blocking_fn, *args))
    while True:
        try:
            # shield로 감싸 wait_for 타임아웃 시에도 작업이 취소되지 않게 한다.
            result = await asyncio.wait_for(asyncio.shield(task), timeout=interval)
            yield ("result", result)
            return
        except asyncio.TimeoutError:
            yield ("keepalive", None)

async def event_generator(query: str):
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
        resp = await asyncio.to_thread(requests.get, search_url, headers=headers, timeout=10)

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

            # 매 포스트마다 진행 상태 1건 전송(프론트 표시 + 스트림 활성화)
            yield "event: progress\ndata: " + json.dumps(
                {"step": 2, "message": f"{idx}/{len(items)}번째 분석 중..."}, ensure_ascii=False) + "\n\n"

            # 1. 크롤링 (Selenium + OCR, 수십 초 블로킹) — 진행 중 keepalive로 스트림 유지
            async for _kind, _payload in _await_with_keepalive(scrape_single_post, link):
                if _kind == "keepalive":
                    yield ": keepalive\n\n"
                else:
                    scraped_title, scraped_content, ocr_texts, image_urls, thumbnail_url = _payload

            if "추출 실패" in scraped_title:
                # 크롤링 오류 등으로 파싱이 막힌 경우 API 검색에 담겨온 제목을 fallback으로 클리닝하여 사용
                clean_title = raw_title.replace("<b>", "").replace("</b>", "") if isinstance(raw_title, str) else "블로그 글"
                scraped_title = clean_title if clean_title.strip() else "블로그 글"
                scraped_content = item.get("description", "").replace("<b>", "").replace("</b>", "")
                ocr_texts = []
                image_urls = []
                thumbnail_url = ""
                if not scraped_content.strip():
                    print(f"[API] ({idx}/{len(items)}) 건너뜀 (크롤링 실패 및 본문 없음)")
                    continue

            # 2. 모델 예측 (블로킹) — 동일하게 keepalive로 스트림 유지
            async for _kind, _payload in _await_with_keepalive(predict_ad, scraped_title, scraped_content):
                if _kind == "keepalive":
                    yield ": keepalive\n\n"
                else:
                    prob, gf = _payload

            # 3. 소프트 확률 결합
            fusion_result = fusion.analyze(prob, scraped_content, ocr_texts, image_urls)
            viral_score = fusion_result['viral_score']
            quick_reason = fusion_result['summary']

            # 4. 카드 데이터 전송 양식 세팅
            card_data = {
                "title": scraped_title,
                "link": link,
                "body": scraped_content[:150] + "...",
                "bloggerName": blogger_name,
                "postDate": post_date,
                "viralScore": viral_score,
                "reason": quick_reason,
                "thumbnail": thumbnail_url,
                "ocrTexts": ocr_texts[:5]
            }

            print(f"[API] ({idx}/{len(items)}) 분석 완료 및 전송: {scraped_title} (점수: {viral_score}점)")
            # SSE 형식에 맞게 yield
            yield f"event: card\ndata: {json.dumps(card_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)

        # Step 3: 모든 작업 완료
        yield "event: progress\ndata: " + json.dumps({"step": 3, "message": "결과 저장 중..."}) + "\n\n"

    except Exception as e:
        print(f"[API ERROR] 스트림 처리 중 에러 발생: {e}")
        yield "event: progress\ndata: " + json.dumps({"step": 3, "message": f"오류 발생: {e}"}) + "\n\n"

@app.get("/api/v1/analyze/stream")
def analyze_stream(query: str = Query(..., description="검색어")):
    print(f"[API] 실시간 SSE 스트리밍 요청 수신: {query}")
    return StreamingResponse(
        event_generator(query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
