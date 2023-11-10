# Use the NVIDIA PyTorch container image as the base image. Works for ubuntu 22.04.3 LTS with GeForce RTX 3070
FROM nvcr.io/nvidia/pytorch:22.07-py3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file from the host to the container
COPY requirements.txt .

# fixes some gpu problems
# RUN apt-get update && apt-get install libgl1 -y

# Install the Python packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose port 6006 for tensorboard
EXPOSE 6006

# Specify the command to run when the container starts
CMD ["bash"]
