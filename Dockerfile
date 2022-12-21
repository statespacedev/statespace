FROM ubuntu:focal
LABEL Description="build environment"

ENV HOME /tmp
SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update -qy

RUN apt-get -qq install -y python3.8 python3-setuptools python3-venv python3-pip python3-tk

COPY statespace ${HOME}/statespace
COPY requirements.txt setup.py README.md MANIFEST.in LICENSE ${HOME}/

RUN cd ${HOME} && \
    python3 setup.py build_ext && \
    python3 setup.py build_py && \
    python3 setup.py sdist && \
    pip3 install --editable . && \
    pytest statespace
