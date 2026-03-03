FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev libffi-dev curl ca-certificates && \
    apt-get clean

# Ensure certificate authority store is up to date
RUN update-ca-certificates

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Copy and set permissions for start script
COPY start.sh .
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Start both backend and Discord bot
CMD ["./start.sh"]