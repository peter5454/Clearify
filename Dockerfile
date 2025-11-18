# ------------------------------------------------------------
# Stage 1: Build Stage (Used to download and cache models)
# ------------------------------------------------------------
FROM python:3.11-buster AS model_builder

# Install ALL necessary components for pre-loading
# 'Dbias' brings in TensorFlow. We manually install PyTorch here.
# Note: You can skip 'Dbias' here and only install the specific classes
# but installing it ensures its dependencies are met for the download step.
RUN pip install transformers Dbias torch tf-keras

# Run a Python script to force the download of the required models.
# We will use the PyTorch class for the PyTorch models (pe5tr/*) and the
# TensorFlow class for the TF model (d4data/*).

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification; \
AutoModelForSequenceClassification.from_pretrained('pe5tr/political_model'); \
AutoModelForSequenceClassification.from_pretrained('pe5tr/sbic_model'); \
AutoModelForSequenceClassification.from_pretrained('pe5tr/fake_news_model'); \
TFAutoModelForSequenceClassification.from_pretrained('d4data/bias-detection-model')"

# Set the cache directory for the models
ENV HF_HOME=/root/.cache/huggingface
# ------------------------------------------------------------
# Stage 2: Final Image (Keep this section clean)
# ------------------------------------------------------------
FROM python:3.11-buster

# Install app dependencies (same as before)
WORKDIR /app
COPY requirements.txt .
# This RUN pip install will re-install torch and Dbias, but it ensures all
# application dependencies are installed correctly for the final stage.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the cached models from the build stage. This is the crucial step.
COPY --from=model_builder $HF_HOME $HF_HOME

# Copy the rest of your application code
COPY . .

# Set Gunicorn Command
ENV PORT 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]