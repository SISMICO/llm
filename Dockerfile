# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent buffering issues with logs
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any are needed later, e.g., for specific libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install pip requirements
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Copy the rest of the application code
COPY ./src ./src
COPY ./.env ./.env
#COPY ./sample_docs ./sample_docs # Copy sample docs if needed inside the image, or use volumes

# Make port 80 available (if you were building a web app, not strictly needed for scripts)
# EXPOSE 80

# Default command (can be overridden in docker-compose.yml)
# This keeps the container running if needed, otherwise it will exit after the script runs.
# For script execution, you'll typically override this.
CMD ["tail", "-f", "/dev/null"]
