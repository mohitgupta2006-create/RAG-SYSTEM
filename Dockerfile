FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CACHE_DIR=/app/knowledge_base_cache \
    HF_HOME=/home/user/.cache/huggingface

COPY requirements.txt requirements-core.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/requirements.txt \
    && useradd -m -u 1000 user \
    && mkdir -p /app/knowledge_base_cache \
    && chown -R user:user /app /home/user

WORKDIR /app

COPY --chown=user:user . /app
USER user

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
