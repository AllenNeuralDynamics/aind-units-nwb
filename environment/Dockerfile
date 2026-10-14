# hash:sha256:7918739ceee6f02e0849b57d17eb05a3ea6e9cb13d7a30b32beb0e4f3ea39d4c
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
    aind-nwb-utils==0.2.9 \
    hdmf==6.1.0 \
    hdmf-zarr==0.13.0 \
    log-schema==0.2.10.dev1 \
    neuroconv==0.9.3 \
    pynwb==4.0.0 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.105.2 \
    wavpack-numcodecs==0.2.2
