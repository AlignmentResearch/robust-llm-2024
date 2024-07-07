FROM pytorch/pytorch as base

ARG DEBIAN_FRONTEND=noninteractive

# Install some useful packages
RUN apt update && \
    apt install -y rsync git vim tini wget curl inotify-tools libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg zstd gcc \
    && python3 -m pip install --upgrade --no-cache-dir pip requests

FROM base as prod
ADD . .
RUN python3 -m pip install '.[tensorflow]'

# Install requirements for repo
# note this only monitors the pyproject.toml file for changes
FROM base as dev
COPY pyproject.toml /workspace/
WORKDIR /workspace

RUN mkdir robust_llm \
    && python3 -m pip install -e ".[dev]" \
    && python3 -m pip uninstall -y robust_llm \
    && rmdir robust_llm && rm pyproject.toml

FROM dev as easydev
RUN curl -fsSL https://code-server.dev/install.sh | sh
ENV PATH="/home/coder/.local/bin:${PATH}"
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-pyright.pyright
