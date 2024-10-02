# hash:sha256:27927f022cae9235f5e59b98b53d55a495230222b8034d971c4f457c2e687d86
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
    wavpack-numcodecs==0.2.1

COPY postInstall /
RUN /postInstall
