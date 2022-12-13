ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="1.6.1"

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

# ENV FORCE_CUDA="1"
# ARG DEPLOY_STAMP=unknown
# Install MMCV, MMseg, MMdet
RUN pip install -U openmim && mim install mmcv-full && pip install mmsegmentation==0.29.1 mmdet==2.26.0


RUN git clone https://github.com/hexafarms/openHexa.git /openHexa
RUN mkdir -p /openHexa/data/images/pictures
WORKDIR /openHexa
RUN pip install -r requirements/fastAPI.txt

# RUN mkdir -p fast_api/input/
# CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]