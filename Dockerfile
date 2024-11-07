# Use a lightweight base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary dependencies for dlib
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the dlib repository
RUN git clone https://github.com/davisking/dlib.git

# Build dlib from source
RUN cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release && \
    cd ..

# Install the Python bindings for dlib
RUN cd dlib && python3 setup.py install

# Copy project files
COPY . .

# Expose the port
EXPOSE 5000

# Run the backend service
CMD ["python", "backend.py"]

