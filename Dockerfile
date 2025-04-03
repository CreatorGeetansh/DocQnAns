# Dockerfile

# 1. Base Image
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Set Python Path early, it's needed for the final CMD
    PYTHONPATH=/app

# 2. System Dependencies (as root)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-eng \
       tesseract-ocr-hin \
       # Add other language packs based on your TESSERACT_LANG env var needs
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Working Directory (as root)
WORKDIR /app

# 4. Install Python Dependencies (as root)
# Copy requirements first for caching
COPY requirements.txt .
# Run pip as root - installs globally in the container's site-packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Create Non-Root User (as root)
# Create user *after* package installation
RUN addgroup --system app \
    && adduser --system --ingroup app app

# 6. Copy Application Code (as root, but set ownership)
# Copy the rest of the application code
# Use --chown to set the owner to the 'app' user we just created
COPY --chown=app:app . .

# 7. Switch to Non-Root User
# Switch user *before* running the application
USER app

# 8. Expose Port (doesn't matter if before/after USER switch)
EXPOSE 8000

# 9. Run Application (as app user)
# CMD will run as the 'app' user because of the preceding 'USER app' line
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]