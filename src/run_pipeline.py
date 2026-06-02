import json
from preprocessing import BlogPreprocessor

def run_bulk_preprocessing(input_file, output_file):
    preprocessor = BlogPreprocessor()
    processed_data = []

    print(f"[INFO] '{input_file}' 읽는 중...")

    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_posts = json.load(f)
    else:
        print("[ERROR] 지원하지 않는 파일 형식입니다. (.json 권장)")
        return

    print(f"[INFO] 총 {len(raw_posts)}개의 데이터 전처리 시작...")
    for post in raw_posts:
        try:
            # 5개의 파라미터 모두 전달 (is_my_money, detected_reason 포함)
            result = preprocessor.process(
                title=post.get('title', ''),
                body=post.get('body', ''),
                label=post.get('label', None),
                is_my_money=post.get('is_my_money', False),
                detected_reason=post.get('detected_reason', [])
            )
            processed_data.append(result)
        except Exception as e:
            print(f"[SKIP] 에러 발생으로 건너뜁니다: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[OK] 전처리 완료! 결과 저장됨: {output_file}")

if __name__ == "__main__":
    run_bulk_preprocessing("blog_data.json", "output_data.jsonl")