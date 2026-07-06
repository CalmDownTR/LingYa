FROM python:3.12-slim

WORKDIR /app

# Install Node.js for web UI build (Debian bookworm ships Node 18+)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install Python dependencies
RUN pip install uv && uv sync

# Build web UI
RUN cd web && npm install && npm run build

EXPOSE 8765

# Start daemon in foreground
CMD ["uv", "run", "python", "main.py"]
