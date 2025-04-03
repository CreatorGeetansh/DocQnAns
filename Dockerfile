# Dockerfile

# 1. Base Image: Use an official Python image.
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Set the Python Path to include the app directory so imports work correctly
    PYTHONPATH=/app

# 2. System Dependencies: Install Tesseract OCR and language packs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-eng \
       tesseract-ocr-hin \
       # Add other language packs based on your TESSERACT_LANG env var needs
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Application Setup
# Create a non-root user
RUN addgroup --system app \
    && adduser --system --ingroup app app

# Create working directory
WORKDIR /app

# Switch to the non-root user
USER app

# 4. Install Python Dependencies
# Copy requirements first for caching
COPY --chown=app:app requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
# Copy the entire project context (including 'backend', 'utils' dirs)
COPY --chown=app:app . .

# 6. Expose Port
EXPOSE 8000

# 7. Run Application
# IMPORTANT: Change 'main:app' to 'backend.main:app'
# This tells uvicorn to look inside the 'backend' directory (which is now a Python package
# because we copied everything to /app and set PYTHONPATH) for the 'main.py' file,
# and find the FastAPI instance named 'app' within that file.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]