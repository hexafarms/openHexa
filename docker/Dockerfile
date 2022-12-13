ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="1.6.1"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

USER root
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# ENV FORCE_CUDA="1"
# ARG DEPLOY_STAMP=unknown
# Install MMCV, MMseg, MMdet
RUN git clone https://github.com/hexafarms/openHexa.git /openHexa
WORKDIR /openHexa

RUN adduser --system hexa 
USER hexa

ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

RUN pip install -r requirements/fastAPI.txt

USER root
RUN mkdir /openHexa/images
RUN mkdir /openHexa/weights
RUN mkdir /openHexa/meta

RUN chown -R hexa /openHexa 

USER hexa