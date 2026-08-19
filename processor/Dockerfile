FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# PyStemmer (fastembed dep) compiles from source on aarch64; use CPU-only torch.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

CMD ["python", "-u", "main.py"]
