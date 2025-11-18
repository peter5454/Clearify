# Use a Python base image, often slim is preferred for smaller size
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file(s)
# Assuming you have a requirements.txt, or list your dependencies here
COPY requirements.txt .

# Install system dependencies (if any, typically for ML/PyTorch)
# PyTorch often requires a full environment or specific libs, but starting simple:
# If you run into issues, you may need to switch to python:3.11-buster or install libgomp, etc.

# Install Python dependencies, including PyTorch/Hugging Face libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Cloud Run requires that your container listens on the PORT environment variable
# The PORT variable is injected by Cloud Run.
# The default port in your code is 8080, which is a good default.
ENV PORT 8080

# Expose the port (optional, but good practice)
EXPOSE 8080

# The command to run the application
# Your main.py uses the Cloud Run standard: `python main.py`
CMD ["python", "main.py"]