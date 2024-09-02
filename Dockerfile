FROM ubuntu:20.04

WORKDIR /app

COPY requirements.txt /app

RUN apt -y update \
    && apt install -y python3-pip net-tools nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r ./requirements.txt \
    && rm -rf /root/.cache/pip