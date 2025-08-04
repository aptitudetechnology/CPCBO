FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app/src

# Create data and results directories
RUN mkdir -p /app/data /app/results /app/logs

# Expose port for dashboard/API
EXPOSE 8050

# Default command
CMD ["python", "src/main.py", "--experiment", "full", "--environment", "production"]
