# Use an official Python runtime as a parent image
# 'slim' version is chosen to reduce image size while maintaining functionality
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir is used to keep the image lean
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create directories for artifacts if they don't exist
RUN mkdir -p data models figures

# The command to run the entire pipeline
# Sequence: Data Generation -> Model Training -> Evaluation
# Using 'sh -c' to allow sequential execution of multiple python scripts
CMD ["sh", "-c", "python src/generate_data.py && python src/train_model.py && python src/evaluate.py"]