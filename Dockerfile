# Use a base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies for dlib and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python1.71.0 \
    libboost-thread1.71.0 \
    libboost-system1.71.0 \
    && apt-get clean

# Install dlib first using pip
RUN pip install --no-cache-dir dlib==19.22.1

# Copy requirements file and install Python dependencies (excluding dlib)
COPY requirements.txt .
RUN sed -i '/dlib/d' requirements.txt  # Remove dlib from requirements.txt if listed
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 5000

# Run the application
CMD ["python", "./app.py"]

