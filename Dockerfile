FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY ./requirements.txt monotonic_align ./

RUN apt update -yqq && \
    apt install -y build-essential && \
    pip install -r requirements.txt && \
    pip install pyopenjtalk monotonic_align
