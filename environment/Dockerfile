# hash:sha256:41d532dc8edf9b322fe588fa4b38b5182ed4454768365f1abd9545732c34e7c9
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.6.0 \
    neuroconv==0.4.6 \
    pynwb==2.6.0 \
    spikeinterface[full]==0.100.0 \
    wavpack-numcodecs==0.1.5
