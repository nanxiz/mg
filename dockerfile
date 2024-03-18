FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

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
