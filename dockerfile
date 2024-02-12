
# # tensorflow use CPU
# FROM tensorflow/tensorflow:latest

# tensorflow use GPU
# "docker run" requires "--gpus" option 
FROM tensorflow/tensorflow:latest-gpu 

# copy python requirements.txt file
COPY requirements.txt /root/

# installing python modules
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r /root/requirements.txt
