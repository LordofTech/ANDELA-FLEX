FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-pharm.txt .
RUN pip install --no-cache-dir -r requirements-pharm.txt

COPY tasks ./tasks

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "tasks.geometric_pharmacophore.align"]
