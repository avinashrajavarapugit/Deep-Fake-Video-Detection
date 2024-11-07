# Use a lightweight base image with Python
FROM python:3.9-slim
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libgtk-3-dev \
    build-essential \
    cmake \
    scikit-image \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    cmake \
    dlib \
    face_recognition
    
# Set the working directory
WORKDIR /app


# Copy project files
COPY . .

# Expose the port
EXPOSE 5000

# Run the backend service
CMD ["python", "backend.py"]

