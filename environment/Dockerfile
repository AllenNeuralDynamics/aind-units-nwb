# hash:sha256:3ae502f99eeb8d5cbd9b5a2d0319f4d8622281f637eb0bb0b7332835258dacd9
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.7.0 \
    pynwb==2.8.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
