# FROM python:3.7.17
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
RUN mkdir /backend
WORKDIR /backend
COPY ./requirements.txt /backend/requirements.txt
RUN apt-get clean && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
    unzip bzip2 \
    openssl libssl-dev \
    libglfw3-dev libgles2-mesa-dev \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl wget \
    ca-certificates \
    locales \
    bash \
    sudo \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "server.py"]