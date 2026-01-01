# Dockerfile for Construction Plan Extraction API
FROM python:3.11-slim

# Install system dependencies (poppler for PDF conversion)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5050

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "--timeout", "300", "app:app"]
