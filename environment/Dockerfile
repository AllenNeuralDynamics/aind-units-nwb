# hash:sha256:45ace56c1c9ff5ab440ac4318d0b2347b392312563155dffad9ada7dab968757
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
    neuroconv==0.7.0 \
    pynwb==2.8.3 \
    spikeinterface[full]==0.102.1 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
