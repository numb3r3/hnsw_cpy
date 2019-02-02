FROM python:3.6.8

MAINTAINER han.xiao@tencent.com

RUN apt-get -y update && \
    apt-get clean && \
    apt-get install nano less && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pip setuptools -U

ENV PIP_FIND_LINKS "http://pypi.open.oa.com/simple/"
ENV PIP_TRUSTED_HOST "pypi.open.oa.com"
