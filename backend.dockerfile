FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install first
COPY requirements_backend.txt /app/requirements_backend.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt

# Create the src directory structure
RUN mkdir -p /app/src/ml_ops_project

# Copy files preserving structure
COPY src/ml_ops_project/api.py /app/src/ml_ops_project/api.py
COPY src/ml_ops_project/model.py /app/src/ml_ops_project/model.py
COPY src/ml_ops_project/__init__.py /app/src/ml_ops_project/__init__.py
COPY outputs/2026-01-21/12-11-02/cnn_model.pth /app/cnn_model.pth

# Create empty __init__.py if it doesn't exist
RUN touch /app/src/__init__.py

EXPOSE 8080

# Run from /app so imports work
CMD exec uvicorn src.ml_ops_project.api:app --host 0.0.0.0 --port 8080
