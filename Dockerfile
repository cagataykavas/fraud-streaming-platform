FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY case_triage.py ./case_triage.py
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
