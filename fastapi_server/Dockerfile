# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Update and upgrade packages to patch vulnerabilities
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Copy the redeploy script and make it executable
COPY redeploy.sh .
RUN chmod +x redeploy.sh

# Expose the port the app runs on
EXPOSE 8000

# Run the application. Note: reload is disabled for production.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
