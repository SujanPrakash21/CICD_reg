# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables for Uvicorn
ENV UVICORN_CMD="uvicorn app:app --host 0.0.0.0 --port 8000"

# Run FastAPI app
CMD ["sh", "-c", "$UVICORN_CMD"]
