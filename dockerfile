
# # tensorflow use CPU
# FROM tensorflow/tensorflow:latest

# tensorflow use GPU
# NOTE: 
#   Two requirements to enable GPU in containers.
#   1. GPU drivers and CUDA are installed on the host OS.
#   2. Use "--gpus" option with "docker run".
FROM tensorflow/tensorflow:latest-gpu 

# copy python requirements.txt file
COPY requirements.txt /root/

# installing python modules
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r /root/requirements.txt
