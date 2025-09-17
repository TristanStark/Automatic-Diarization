# Étape 1: runtime Python slim
FROM python:3.11-slim

# ffmpeg (transcodage)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Code
COPY diarization.py ./
COPY .env.sample ./

# Par défaut: .env lu par python-dotenv dans le script
ENV PYTHONUNBUFFERED=1

# Cmd par défaut (modifiable)
CMD ["python", "diarization.py"]
