# ------------------------------------------------------------
# Stage 1: Final Image
# ------------------------------------------------------------
# We only need one stage now, as model downloading happens at runtime.
FROM python:3.11-slim

# Install necessary system packages if needed (e.g., for Dbias/TensorFlow)
# python:3.11-slim is based on Debian/Alpine. We'll stick to a common base.
# If you need specific system libraries (like libgomp for some ML libraries), add them here.
# RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/* 

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies.
# Note: requirements.txt MUST now include google-cloud-storage, torch, transformers, and Dbias dependencies.
COPY requirements.txt .
# Use --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the environment variable for the temporary directory.
# This is where your GCS code will download the models to.
# Note: Cloud Run usually makes /tmp available, but this is a good explicit practice.
ENV LOCAL_MODEL_BASE_PATH="/tmp/huggingface_models"

# Set Gunicorn Command
ENV PORT 8080
# Your main application will run, and the import of ml_analysis.py 
# will trigger the model download from GCS *before* gunicorn serves requests.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]