# Dockerfile for Image Analyzer - Railway Deployment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - libgl1-mesa-glx: OpenCV GUI support (headless)
# - libglib2.0-0: OpenCV dependencies
# - tesseract-ocr: OCR support
# - libtesseract-dev: Tesseract development libraries
# - build-essential: For compiling Python packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first for better caching
COPY requirementsDeployment.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirementsDeployment.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs

# Expose port (Railway will set PORT env var)
EXPOSE 5000

# Health check (optional - Railway has its own health checks)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:${PORT:-5000}/login', timeout=5)" || exit 1

# Start command with Gunicorn
CMD gunicorn --worker-class eventlet --workers 1 --bind 0.0.0.0:${PORT:-5000} --timeout 120 app:app

