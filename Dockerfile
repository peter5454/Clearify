# Use a more robust Debian-based image for stability with ML libraries
FROM python:3.11-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Cloud Run requires that your container listens on the PORT environment variable
ENV PORT 8080

# Expose the port (optional)
EXPOSE 8080

# The command to run the application using the production WSGI server, Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]