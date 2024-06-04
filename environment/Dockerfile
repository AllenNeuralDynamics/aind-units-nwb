<<<<<<< HEAD
# hash:sha256:da7eaffb5f0df51bee555229c66eda00dca73e4b86b19a17d0f6dda152b0c1a8
=======
# hash:sha256:12c76ac996ce23a21f7ada44c598d881ad3e6ac6e10dfc93af745154de212366
>>>>>>> local_dispatch
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.6.0 \
    pynwb==2.6.0 \
<<<<<<< HEAD
=======
    spikeinterface[full]==0.100.5 \
>>>>>>> local_dispatch
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
