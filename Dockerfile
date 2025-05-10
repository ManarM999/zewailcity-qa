# Use a minimal base image with Python
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variables (optional if Railway handles $PORT)
ENV PORT=5000

# Expose port
EXPOSE $PORT

# Run the Flask app with Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
