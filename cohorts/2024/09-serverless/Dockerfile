# Use the existing image as the base
FROM agrigorev/model-2024-hairstyle:v3

COPY requirements.txt .
# Copy your lambda function code into the image
COPY lambda_handler.py /var/task/

# Set the entry point for the lambda function
RUN pip install -r requirements.txt
RUN pip install numpy==1.23.1


# Set the entry point for the lambda function
# ... existing code ...

# Set the entry point to run the lambda_handler.py script
ENTRYPOINT ["python", "/var/task/lambda_handler.py"]
