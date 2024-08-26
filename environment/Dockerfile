# hash:sha256:b67009bf48d53070968521e6ecf2b25a1fc509a6e0f94771c9219b49764dbdbc
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.8.0 \
    pynwb==2.8.1 \
    spikeinterface[full]==0.101.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
