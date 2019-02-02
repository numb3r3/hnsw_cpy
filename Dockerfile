FROM python:3.6.8

MAINTAINER han.xiao@tencent.com

RUN apt-get -y update && \
    apt-get clean && \
    apt-get -y install nano less locales && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip install pip setuptools -U

ENV PIP_FIND_LINKS "http://pypi.open.oa.com/simple/"
ENV PIP_TRUSTED_HOST "pypi.open.oa.com"
