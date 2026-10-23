FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY artifacts/model.joblib ./artifacts/model.joblib
RUN pip install --no-cache-dir .
ENV MODEL_PATH=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["uvicorn", "housing_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
