# Base image
FROM python:3.11.5-slim
WORKDIR /app
COPY ["model1.bin", "dv.bin", "./"]

# Set the working directory
WORKDIR /app

# Copy requirements.txt generated locally
COPY requirements.txt ./

# Install dependencies using pip
RUN pip install -r requirements.txt

# Copy application files
COPY app.py dv.bin model1.bin ./

# Expose the application port
EXPOSE 9696

# Start the application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:9696", "app:app"]
