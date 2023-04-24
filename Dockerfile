FROM python:3.8-bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libmariadb-dev \
    openjdk-11-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install same packages
RUN pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --no-compile \
      --upgrade \
      install torchserve==0.7.1 torch==1.13.1 captum==0.6.0 PyYAML==6.0 tokenizers==0.13.2 transformers==4.25.1 torchtext==0.14.1

# Copy default configurations
COPY config.properties config.properties