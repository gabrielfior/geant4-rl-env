# Locked-down runner for model-written Geant4 scripts. Datasets are mounted read-only at /g4data at run time.
FROM --platform=linux/amd64 python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 libx11-6 libxext6 libice6 libsm6 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir geant4_pybind numpy \
 && useradd -m -u 1000 runner
USER runner
WORKDIR /work
