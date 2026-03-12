FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && apt-get -y install git

COPY data/ data/

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY src/ src/

COPY main.py main.py

COPY accentuator_accuracy.py accentuator_accuracy.py

COPY tests.py tests.py
