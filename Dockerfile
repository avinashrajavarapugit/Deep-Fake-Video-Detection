# Use a lightweight base image with Python
FROM python:3.9-slim
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libgtk-3-dev \
    libx11-dev \
    libopenblas-dev \
    liblapack-dev \
    build-essential \
    cmake 

# Install Python packages
RUN pip3 install cmake

RUN pip3 install dlib
    
# Set the working directory
WORKDIR /app


# Copy project files
COPY . .

# Expose the port
EXPOSE 5000

# Run the backend service
CMD ["python", "backend.py"]

