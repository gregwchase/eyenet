# Set the NVIDIA-MXNet image as default
FROM nvcr.io/nvidia/mxnet

# Clone the EyeNet repository
RUN git clone https://github.com/gregwchase/dsi-capstone.git

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME EyeNet
