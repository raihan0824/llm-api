# Use an image that has CUDA pre-installed
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

# Set the working directory
WORKDIR /opt/app

# Copy the requirements file and install the requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the vllm-random-seed directory and install it
COPY ./vllm-random-seed ./vllm-random-seed
RUN cd /opt/app/vllm-random-seed &&\
     pip install --no-deps --no-build-isolation .

# Copy the rest of the application
COPY . ./

# Expose port 8000
EXPOSE 8000

# Set the command to run the application
CMD ["python3", "main.py"]