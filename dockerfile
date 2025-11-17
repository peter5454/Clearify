# Use a minimal Python base image
FROM python:3.11-slim

# Set environment variables for the application
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies (Flask, Vertex SDK, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# If you use spacy, you might need to download a model:
# RUN python -m spacy download en_core_web_sm

# Copy the entire application code and frontend files
COPY . $APP_HOME

# Cloud Run expects the app to listen on the port specified by the PORT environment variable
# The default is 8080.
ENV PORT 8080

# Run the application using Gunicorn (a production-ready WSGI server)
# Ensure 'main:app' matches your Flask app object (app = Flask(__name__) in main.py)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app