# hash:sha256:bf633d289340ee3c4ed7748a0b4d428cea7016e095bb5d5eafe75a3a2a1b4dec
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.8.0 \
    neuroconv==0.6.0 \
    pynwb==2.8.1 \
    spikeinterface[full]==0.101.1 \
    wavpack-numcodecs==0.1.5
