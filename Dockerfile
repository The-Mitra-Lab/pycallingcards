FROM rocker/r-ubuntu

RUN  apt-get update
RUN  apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    wget \
	build-essential \
    python3.9 \
	python3-pip

# Clean up
RUN apt-get autoremove -y

RUN pip install --upgrade pip

RUN pip install "git+https://github.com/The-Mitra-Lab/pycallingcards.git" --upgrade
