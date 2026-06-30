# hash:sha256:481c47873c756f85dbd74865c2aa27d53fc41e1f6ca2f617d3ed57e26bbbe5ed
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    aind-nwb-utils==0.2.5 \
    hdmf==6.1.0 \
    hdmf-zarr==0.13.0 \
    neuroconv==0.9.3 \
    pynwb==4.0.0 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.104.7 \
    wavpack-numcodecs==0.2.2
