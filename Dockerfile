FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ft_engineering.py src/model_deploy.py src/model.joblib ./
EXPOSE 8000
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
