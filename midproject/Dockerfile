# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

COPY requirements.txt /app/
# Install dependencies
RUN pip install -r requirements.txt

# Copy files
COPY train.py predict.py best_heating_load_model_pipeline.pkl   /app/



# Expose port
EXPOSE 5100

# Run the Flask app
CMD ["python", "predict.py"]
