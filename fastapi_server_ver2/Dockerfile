# 1. Base Image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY . .

# 5. Expose port
# The port the container will listen on. This should match the port in the CMD.
EXPOSE 8000

# 6. Run application
# Use gunicorn to run the app with uvicorn workers for production.
# The number of workers can be adjusted based on the server's CPU cores.
# Example: (2 * CPU_CORES) + 1
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
