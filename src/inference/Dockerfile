FROM python:3.10

ARG model_name=prod_model.pickle
ARG vectorizer_name=prod_vectorizer.pickle
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/processed /app/data/processed

# Copy models folder to Docker
COPY outputs/models/${model_name} /app/outputs/models/${model_name}
COPY outputs/models/${vectorizer_name} /app/outputs/models/${vectorizer_name}

# Copy the code itself
COPY src/inference /app/src/inference

# Copy useful scripts and configs
COPY utils.py /app
COPY src/evaluation.py /app/src/evaluation.py

COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 /app/src/inference/run_inference.py