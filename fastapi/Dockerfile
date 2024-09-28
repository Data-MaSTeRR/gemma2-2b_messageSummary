# 베이스 이미지로 Python 3.9 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 패키지 의존성 복사 및 설치
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 앱 코드 복사
COPY . /app

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
