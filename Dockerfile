FROM python:3.6.8

MAINTAINER han.xiao@tencent.com

WORKDIR /

RUN apt-get -y update && \
    apt-get clean && \
    apt-get install pico && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pip -U
