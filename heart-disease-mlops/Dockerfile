# 1. Base image — a slim Debian with Python 3.11
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Copy ONLY requirements first (Docker layer caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and model
COPY src/ ./src/
COPY best_model.joblib .

# 6. Document the exposed port (metadata only; does not publish it)
EXPOSE 8000

# 7. Start the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
