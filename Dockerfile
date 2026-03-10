FROM python:3.8-slim

WORKDIR /app

COPY data/ data/

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY main.py main.py

COPY src/ src/

COPY tests/ tests/
