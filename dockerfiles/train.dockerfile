FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/

# Install dependencies with uv
RUN uv sync --locked --no-cache --no-install-project

# Set the entrypoint to run the training script
ENTRYPOINT ["uv", "run", "python", "src/ml_ops_project/train.py"]

# COMMANDS FOR BUILDING AND RUNNING THE DOCKER IMAGE:
# docker build -f dockerfiles/train.dockerfile . -t train:latest
# docker run --name experiment1 train:latest (add --rm to the command and it will be deleted after running)
#
# LIST ALL CONTAINERS:
# docker ps -a
#
# REMOVE A CONTAINER:
# docker rm <container_id>
