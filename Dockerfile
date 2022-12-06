FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN  apt-get update && \
     apt-get install -y --no-install-recommends \
       software-properties-common \
       dirmngr \
       wget \
	   build-essential \
     python3.9-dev \
     gcc \
	   python3-pip \
     bedtools \
       git

# Clean up
RUN apt-get autoremove -y

RUN python3.9 -m pip install --upgrade pip

# this is to install d3blocks
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN python3.9 -m pip install "git+https://github.com/cmatKhan/pycallingcards.git@raw_processing" --upgrade
