FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/nanxiz/mg.git /workspace

# COPY requirements.txt /workspace
RUN pip install --no-cache-dir -r requirements.txt

# COPY handler.py /workspace

CMD ["python", "handler.py"]
