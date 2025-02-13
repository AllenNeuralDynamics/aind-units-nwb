# hash:sha256:c4f632984c76005c9a6c46b1952d6acebc49d5dfd1d415c46f9000731c958a24
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    hdmf-zarr==0.11.0 \
    neuroconv==0.6.7 \
    pynwb==2.8.3 \
    spikeinterface[full]==0.102.0 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
