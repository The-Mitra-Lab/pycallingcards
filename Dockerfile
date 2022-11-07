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
       python3.9 \
	   python3-pip \
       git

# Clean up
RUN apt-get autoremove -y

RUN pip install --upgrade pip

RUN pip install "git+https://github.com/The-Mitra-Lab/pycallingcards.git" --upgrade
