FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "identity_verification.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
