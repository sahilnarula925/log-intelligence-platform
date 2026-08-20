FROM python:3.11-slim

WORKDIR /app

COPY ingester/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY ingester ./ingester

ENV PYTHONPATH=/app
CMD ["python", "-u", "ingester/main.py"]
