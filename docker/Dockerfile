# Set the NVIDIA-MXNet image as default
FROM nvcr.io/nvidia/mxnet:17.10


# Set the working directory
WORKDIR /workspace


# Copy the current directory contents into the container at /workspace
ADD . /workspace


# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


# Install the Keras 1.2.2 fork for Keras MXNet
RUN git clone https://github.com/dmlc/keras.git
RUN cd keras && python setup.py install


# Clone the EyeNet repository
RUN git clone https://github.com/gregwchase/dsi-capstone.git


# Make port 80 available to the world outside this container
EXPOSE 80


# Define environment variable
ENV NAME EyeNet
