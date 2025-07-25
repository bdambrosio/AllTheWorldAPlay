### UI build stage
FROM node:20 AS ui
LABEL org.opencontainers.image.source=https://github.com/bdambrosio/AllTheWorldAPlay
WORKDIR /build

# 1.  copy only the manifests first (better cache)
COPY src/sim/webworld/package.json ./
COPY src/sim/webworld/package-lock.json ./

# ──--------------------------------------
# DEBUG – drop after things work
RUN echo "===== build dir contents =====" \
 && ls -al \
 && echo "===== package-lock head =====" \
 && head -10 package-lock.json
# ──--------------------------------------


# 👉 robust npm settings + install
RUN npm config set fetch-retries 5 
RUN npm ci       # or `npm ci` if you want dev deps for the build

# 2.  copy the rest of the UI source
COPY src/sim/webworld/ .

# 3.  produce the static bundle
RUN npm run build              # by default writes to webworld/build
###############################################################
# 2) Runtime stage: CUDA‑12 +   YOUR OWN   Python 3.12 install #
###############################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04  

# --- A. add Python 3.12  ------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# --- B. (optional) build tools if you compile wheels --------
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git \
    pkg-config libcairo2-dev libgirepository1.0-dev gir1.2-gtk-3.0 &&\
    rm -rf /var/lib/apt/lists/*

# Use an official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install build dependencies and Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    procps \   
    htop \        
    nano \         
    net-tools \    
    iputils-ping \ 
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy only the allowed directories from src
COPY src /app/src
COPY src/constraints.txt /app/src/constraints.txt

# Copy the built UI files from the ui stage
COPY --from=ui /build/build ./static

RUN python -m pip install --upgrade pip                \
    && pip config set global.progress_bar off          \
    && pip config set global.timeout 1000              \
    && pip config set install.trusted-host pypi.org \
    && pip config set install.trusted-host files.pythonhosted.org

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libcairo2-dev libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "== /app/src listing ==" && ls -l /app/src | head

RUN pip install --no-cache-dir \
    -r src/requirements.txt \
    -c src/constraints.txt

# Install React dependencies and build
WORKDIR /app/src/sim/webworld
COPY src/sim/webworld/package*.json ./
RUN npm install
COPY src/sim/webworld/ .

# Add src to PYTHONPATH
#ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose the port your app runs on
EXPOSE 8000
EXPOSE 5008
EXPOSE 3000

# Add this before the CMD line:
COPY start.sh /app/start.sh
COPY start-ablation.sh /app/start-ablation.sh
COPY start-Qwen3.sh /app/start-Qwen3.sh
COPY start-mistral.sh /app/start-mistral.sh
COPY start-openai.sh /app/start-openai.sh
RUN chmod +x /app/start.sh /app/start-Qwen3.sh /app/start-mistral.sh /app/start-openai.sh /app/start-ablation.sh

WORKDIR /app
# Replace the CMD line with:
CMD ["/app/start-openai.sh"]