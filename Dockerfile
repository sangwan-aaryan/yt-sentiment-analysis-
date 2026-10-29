FROM python:3.14-slim

ENV HOST=0.0.0.0 \
    PORT=5000 \
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false \
    NLTK_DATA=/usr/local/share/nltk_data \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d "$NLTK_DATA" stopwords wordnet

COPY backend/main.py main.py

EXPOSE 5000

CMD ["python", "main.py"]
