FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN apt-get update \
    && apt-get -y install python \
    python-pip \
    python-dev \
    git vim locales\
    openssh-server

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install image
RUN pip install pillow
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /workspace
ADD . .
ENV PYTHONPATH $PYTHONPATH:/workspace

RUN pip install -r requirements.txt

ENV LANG=ko_KR.UTF-8
RUN locale-gen ko_KR ko_KR.UTF-8
RUN update-locale LANG=ko_KR.UTF-8

RUN chmod -R a+w /workspace

