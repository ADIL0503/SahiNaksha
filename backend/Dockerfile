FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
RUN mkdir -p /app/uploads /app/models

ENV SAHINAKSHA_MODEL_PATH=/app/models/sahinaksha-building-seg.pt
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
