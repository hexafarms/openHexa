ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="1.3.17"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# Install MMSegmentation
ARG DEPLOY_STAMP=unknown

RUN git clone https://github.com/HexaFarms/MMsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
RUN git fetch
RUN git checkout master
ENV FORCE_CUDA="1"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN cd ..
RUN git clone https://github.com/HexaFarms/Hexa_image.git /Hexa_image
RUN mkdir -p /Hexa_image/data/images/pictures
WORKDIR /Hexa_image
RUN pip install -r requirements.txt

# RUN mkdir -p fast_api/input/
# CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]