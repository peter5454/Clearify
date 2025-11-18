# ------------------------------------------------------------
# Stage 1: Build Stage (Used to download and cache models)
# ------------------------------------------------------------
FROM python:3.11-buster AS model_builder

# Install libraries needed to download models
RUN pip install transformers Dbias

# Run a Python script to force the download of the required models
# The models are: pe5tr/political_model, pe5tr/sbic_model, pe5tr/fake_news_model
# And the Dbias model: d4data/bias-detection-model

RUN python -c " \
    from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoModelForSequenceClassification.from_pretrained('pe5tr/political_model'); \
    AutoModelForSequenceClassification.from_pretrained('pe5tr/sbic_model'); \
    AutoModelForSequenceClassification.from_pretrained('pe5tr/fake_news_model'); \
    AutoModelForSequenceClassification.from_pretrained('d4data/bias-detection-model', use_safetensors=True)"

# Set the cache directory for the models
ENV HF_HOME=/root/.cache/huggingface
# ------------------------------------------------------------
# Stage 2: Final Image
# ------------------------------------------------------------
FROM python:3.11-buster

# Install app dependencies (same as before)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the cached models from the build stage
COPY --from=model_builder $HF_HOME $HF_HOME

# Copy the rest of your application code
COPY . .

# Set Gunicorn Command (Required)
ENV PORT 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]