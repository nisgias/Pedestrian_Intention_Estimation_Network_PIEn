FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/project

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir transformers==4.40.0 accelerate timm
RUN pip install --no-cache-dir torchvision==0.17.0 torchaudio==2.2.0

RUN mkdir -p /data/PIE \
             /data/pie_interface \
             /workspace/PIE_PREP_OUT \
             /workspace/models \
             /workspace/project/m2f_cache

CMD ["/bin/bash"]