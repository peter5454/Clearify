# Use a Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies, including PyTorch/Hugging Face libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Cloud Run requires that your container listens on the PORT environment variable
ENV PORT 8080

# Expose the port (optional)
EXPOSE 8080

# The command to run the application
CMD ["python", "main.py"]