FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "rdkit>=2023.9.1" "numpy>=1.24.0"

COPY tasks ./tasks

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "tasks.geometric_pharmacophore.align"]
