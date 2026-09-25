FROM python:3.11-slim

WORKDIR /app

COPY app.py .
COPY video_game_sales_model.joblib .

RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn joblib pydantic

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
