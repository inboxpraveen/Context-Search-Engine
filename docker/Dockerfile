# contextsearch: keyword + semantic document search, self-hosted.
#
#   docker build -t contextsearch -f docker/Dockerfile .
#   docker run -p 8080:8080 -v cse-data:/data -v /srv/docs:/docs:ro \
#     -e CSE_API_KEY=change-me -e CSE_ALLOWED_ROOTS=/docs contextsearch
#
# The image ships the CPU build of torch. For a GPU host, build from an
# NVIDIA CUDA base image and drop the --index-url line.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/models \
    CSE_PROFILE=production \
    CSE_DATA_DIR=/data \
    CSE_HOST=0.0.0.0 \
    CSE_PORT=8080

WORKDIR /app

# opencv (pulled in by RapidOCR) links against these even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first so the dense extra does not pull the CUDA wheels (several GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY contextsearch ./contextsearch
RUN pip install ".[dense,ocr]"

# tests are copied for the optional self-check: docker run --rm contextsearch python -m pytest -q
COPY tests ./tests
RUN pip install pytest httpx

RUN useradd --create-home --uid 1000 contextsearch && mkdir -p /data && chown contextsearch /data
USER contextsearch
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["contextsearch", "serve"]
