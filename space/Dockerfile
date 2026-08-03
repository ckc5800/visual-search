FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY static/ static/
COPY data/index/ data/index/

# HF 캐시(모델·데이터셋 다운로드)는 쓰기 가능한 경로로
ENV HF_HOME=/tmp/hf PYTHONUTF8=1

EXPOSE 7860
CMD ["uvicorn", "--app-dir", "src", "api:app", "--host", "0.0.0.0", "--port", "7860"]
