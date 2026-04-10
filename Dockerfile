FROM python:3.10-slim

WORKDIR /app
LABEL org.opencontainers.image.authors="Pratyush Mishra"
ENV PROJECT_AUTHOR="Pratyush Mishra"

COPY . /app
RUN pip install --no-cache-dir -r requirements-pipeline.txt

CMD ["python", "app.py"]
