# Dynamic Trend & Event Detector
# Python 3.11 — required for Gensim ≥ 4.3 + all other dependencies
FROM python:3.11-slim

WORKDIR /app

# System deps: gcc/g++ for scipy/numpy wheels, libgomp for KMeans parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy English model (NER entity tracker)
RUN python -m spacy download en_core_web_sm

# Copy project files (data/ and reports/ are volume-mounted at runtime)
COPY src/       ./src/
COPY notebooks/ ./notebooks/
COPY web/       ./web/
COPY *.py       ./
COPY *.sh       ./
COPY *.md       ./
RUN chmod +x *.sh

# Expose ports: 8501 Streamlit · 8000 FastAPI
EXPOSE 8501 8000

# Default: show help; override with docker-compose service commands
CMD ["python", "-c", "print('Use docker-compose to run pipeline, dashboard, or api.')"]
