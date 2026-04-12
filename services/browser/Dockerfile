FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

RUN pip install --no-cache-dir fastapi==0.135.3 uvicorn==0.44.0 pydantic==2.12.5

RUN useradd -m -s /bin/bash browser \
    && mkdir -p /home/browser/.cache \
    && chown -R browser:browser /home/browser

USER browser
WORKDIR /home/browser

COPY --chown=browser:browser main.py /home/browser/main.py

EXPOSE 8086

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8086"]
