FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel AS compile

ARG DEBIAN_FRONTEND=noninteractive
# git is required for installing flash_attn.
RUN apt update \
    && apt install -y git \
    && rm -rf /var/lib/apt/lists/*

# To port Python package installs in this stage to other Docker stages, we can
# store the installs in a virtualenv: https://stackoverflow.com/a/61879604/4865149
RUN python3 -m venv /usr/local/venv
ENV PATH="/usr/local/venv/bin:$PATH"

# Install [cuda] packages that require `nvcc`, which is available in `devel`
# PyTorch images and not `runtime` images.
COPY pyproject.toml .
RUN python3 -m pip install --no-cache-dir '.[cuda-deps]'
# CUDA_HOME is required for installing flash_attn.
ENV CUDA_HOME=/usr/local/cuda-12.1
RUN python3 -m pip install --no-cache-dir '.[cuda]'

# We use a runtime image here as the base to cut down the image size by a few
# GB, but we still consider the resulting image `dev` to be a development image
# because it contains convenience installs that aren't strictly needed for
# experiments. For simplicity we're using the same image `dev` for experiments
# and develompent.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS dev

ARG DEBIAN_FRONTEND=noninteractive
# Install some useful packages
RUN apt update \
    && apt install -y rsync git parallel vim tini wget curl inotify-tools libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg zstd gcc \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade --no-cache-dir pip requests

# VSCode code server
RUN curl -fsSL https://code-server.dev/install.sh | sh
ENV PATH="/home/coder/.local/bin:${PATH}"
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-pyright.pyright
RUN code-server --install-extension ms-python.black-formatter
# URL is from https://marketplace.visualstudio.com/items?itemName=GitHub.copilot&ssr=false#version-history
RUN wget https://marketplace.visualstudio.com/_apis/public/gallery/publishers/GitHub/vsextensions/copilot/1.243.1189/vspackage --output-document copilot.vsix.gz \
    && gzip --decompress copilot.vsix.gz \
    && code-server --install-extension copilot.vsix \
    && rm copilot.vsix

# Copy the venv with built packages from compile stage
COPY --from=compile /usr/local/venv /usr/local/venv
ENV PATH="/usr/local/venv/bin:$PATH"

COPY pyproject.toml /workspace/robust-llm/
WORKDIR /workspace

# Install in developer mode with a mock dependency-only project.
# This installs dependencies, and makes `robust_llm` importable from
# `/workspace/robust-llm`. We then remove that directory: the user
# needs to install a real version of the code (e.g. by Git cloning it).
RUN cd robust-llm \
    && mkdir robust_llm \
    && touch robust_llm/__init__.py \
    && python3 -m pip install --no-cache-dir -e ".[dev]" \
    && cd .. \
    && rm -rf robust-llm
