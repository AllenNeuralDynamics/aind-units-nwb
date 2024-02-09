# hash:sha256:56b68d252a4b21fc5741a450d0f466a6749cb31b0cb83797c895033887a1c9f6
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    spikeinterface[full]==0.100.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
