FROM python:3.11-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    "numpy>=1.24.0" \
    "scipy>=1.10.0" \
    "nibabel>=5.0.0" \
    "scikit-image>=0.21.0"

COPY tasks ./tasks

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "tasks.qsm_recon.reconstruct"]
