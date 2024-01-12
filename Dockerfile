# Our base image
FROM tensorflow/tensorflow:latest-gpu-jupyter


RUN apt-get update
RUN apt-get install -y vim 
RUN apt install -y graphviz
RUN apt install -y tmux
RUN bash -c export PATH="/.local/bin:$PATH"


# set up permissions for mfm group
# this should be set in docker-compose.yml
ARG GID
# make the user mfm:mfm where the mfm group # is GID 
RUN groupadd -r -g $GID mfm && useradd --no-log-init -r -m -g $GID mfm
USER mfm:mfm

# Copy the requirements.txt file to our Docker image
# COPY --chown=$USER requirements.txt ./
COPY requirements.txt ./

# Install the requirements.txt
RUN pip install -r requirements.txt

