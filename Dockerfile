# Request Python Slim image
FROM python:3.7.6-slim

# Install libraries
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /eyenet

# Copy files to image
COPY . .
