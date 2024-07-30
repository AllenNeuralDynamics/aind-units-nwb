# hash:sha256:7979344cc6e584177b3073ceca420ec673c7fe0cdd2f01d65c3329566a89abc1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.6.0 \
    neuroconv==0.4.6 \
    pynwb==2.6.0 \
    spikeinterface[full]==0.100.7 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
