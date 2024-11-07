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
    libboost-python-dev \
    libboost-thread-dev \
    libboost-system-dev \
    && apt-get clean

# Install dlib first using 
COPY dlib-19.22.1-cp39-cp39-win_amd64.whl .
RUN pip install ./dlib-19.22.1-cp39-cp39-win_amd64.whl
# Copy requirements file and install Python dependencies (excluding dlib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 5000

# Run the application
CMD ["python", "./app.py"]

