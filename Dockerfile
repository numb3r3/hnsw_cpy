FROM python:3.6.8

MAINTAINER han.xiao@tencent.com

WORKDIR /

RUN apt-get -y update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD . /

RUN pip install pip -U
