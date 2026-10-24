FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Heavy, rarely-changing layer first -> better build caching
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY scripts ./scripts

# Don't run as root
RUN useradd -m runner && mkdir -p /app/runs /app/data && chown -R runner /app
USER runner

ENTRYPOINT ["bash", "scripts/run_pipeline.sh"]
CMD ["-d", "ccpp"]
