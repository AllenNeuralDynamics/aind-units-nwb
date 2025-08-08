# hash:sha256:758707cce53bc2e855006be9ca9a190b81b7548e5421e545ea8ab38c719e0998
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
    neuroconv==0.7.1 \
    pynwb==2.8.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.103.0 \
    wavpack-numcodecs==0.2.2
