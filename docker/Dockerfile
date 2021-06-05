# Docker file for a slim Ubuntu-based Python3 image

FROM ubuntu:latest
MAINTAINER fnndsc "dev@babymri.org"

RUN apt-get update \
  && apt-get install -y locales \
  && apt-get install -y libpq-dev \
  && apt-get install -y python3-pip python3-dev \
  && python3 -m pip install --upgrade pip \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip3 install cython
# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
COPY . /home
RUN pip3 install /home/emd_lib/
RUN pip3 install /home/text-preprocessing/
RUN pip3 install -r /home/requirements.txt
RUN apt-get install -y git
RUN apt-get install wget
# RUN init.sh

ENTRYPOINT ["/bin/bash"]
# cmd ["bash"]