FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.135.3 \
    uvicorn==0.44.0 \
    pydantic==2.12.5 \
    httpx==0.28.1 \
    pandas==2.2.3 \
    numpy==2.4.4

RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /home/sandbox

COPY --chown=sandbox:sandbox main.py /home/sandbox/main.py

EXPOSE 8087

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8087"]
