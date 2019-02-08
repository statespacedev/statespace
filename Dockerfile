FROM python:3.6-slim

COPY requirements.txt /
RUN pip install -r requirements.txt
COPY . /

