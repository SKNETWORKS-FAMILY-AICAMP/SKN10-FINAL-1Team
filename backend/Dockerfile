# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Update and upgrade packages to patch vulnerabilities, install git
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Create directory for static files and media
RUN mkdir -p /app/staticfiles /app/media

# Copy and set executable permission for startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["/app/start.sh"] 