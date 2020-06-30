FROM ubuntu:18.04

MAINTAINER Giuseppe Capaldi "giuppocapaldi@gmail.com"

ENV DEBIAN_FRONTEND="noninteractive"
# Install required packages
RUN apt-get update -y && \
    apt-get install -y python3.6 python3-pip python3-dev \
    bzip2 wget jupyter

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

 
# Install required python packages
RUN pip3 install -r requirements.txt

COPY . /app
# Run python modules
# RUN python3 custom_gym/scenario_objects.py
# RUN python3 custom_gym/plotting.py
# RUN python3 custom_gym/env_wrapper.py

 # Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Command that starts up the notebook 
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]