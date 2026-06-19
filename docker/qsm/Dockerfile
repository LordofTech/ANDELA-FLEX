FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements-qsm.txt .
RUN pip install --no-cache-dir -r requirements-qsm.txt

COPY tasks ./tasks

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "tasks.qsm_recon.reconstruct"]
