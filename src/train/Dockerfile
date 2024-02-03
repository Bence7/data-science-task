# Dockerfile for training
# Start from a base Python image
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy the code itself
COPY src/train /app/src/train
COPY data /app/data
COPY src/data_preparation.py /app/src/data_preparation.py
COPY src/evaluation.py /app/src/evaluation.py
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 /app/src/train/train.py