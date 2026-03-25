# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG REQUIREMENTS_FILE=requirements.txt
ARG TORCH_WHEEL_INDEX_URL=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

# Marker relies on torch/image tooling. These base libraries keep the image lean
# while still being compatible with common CPU-only runs.
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org/debian|https://${APT_MIRROR}/debian|g; s|http://security.debian.org/debian-security|https://${APT_MIRROR}/debian-security|g" /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://deb.debian.org/debian|https://${APT_MIRROR}/debian|g; s|http://security.debian.org/debian-security|https://${APT_MIRROR}/debian-security|g" /etc/apt/sources.list; \
    fi; \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements /app/requirements

RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    pip install -r /app/${REQUIREMENTS_FILE}; \
    if [ -n "${TORCH_WHEEL_INDEX_URL}" ]; then \
        TORCH_VERSION="$(pip show torch 2>/dev/null | sed -n 's/^Version: //p' | sed 's/+.*$//' | head -n 1)"; \
        TORCHVISION_VERSION="$(pip show torchvision 2>/dev/null | sed -n 's/^Version: //p' | sed 's/+.*$//' | head -n 1)"; \
        if [ -n "${TORCH_VERSION}" ]; then \
            TORCH_SPECS="torch==${TORCH_VERSION}"; \
            if [ -n "${TORCHVISION_VERSION}" ]; then \
                TORCH_SPECS="${TORCH_SPECS} torchvision==${TORCHVISION_VERSION}"; \
            fi; \
            pip install --index-url "${TORCH_WHEEL_INDEX_URL}" --extra-index-url "${PIP_INDEX_URL}" ${TORCH_SPECS}; \
        fi; \
    fi

COPY . /app

CMD ["python", "src/01-structure_aware_chunking/pipeline.py", "--help"]
