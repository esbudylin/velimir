FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY data/ data/

COPY main.py main.py

COPY src/ src/

COPY tests/ tests/
