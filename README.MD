# OCR Server with Gemini API

닥터도리 약봉투 OCR 서버

## 기능
- 약봉투 이미지 분석
- 한글 → 영어 번역
- Gemini 2.0 Flash 사용

## 설치
```bash
pip install -r requirements.txt
```

## 실행
```bash
uvicorn main:app --reload
```

## 테스트
- 로컬: http://localhost:8000
- 엔드포인트: POST /ocr

## 환경변수
`.env` 파일 필요:
```
GEMINI_API_KEY=your_api_key_here
```

## 배포
Render 무료 플랜 사용