# Backend image for Railway: FastAPI + TensorFlow (CPU) serving app_model/model.keras
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app.py server.py ./
COPY app_model ./app_model

# Railway injects $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
