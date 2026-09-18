# Deployment

contextsearch is one process with one data folder. This page covers a production setup on Linux, the Windows equivalent, Docker, a reverse proxy, backups, and moving up from version 2.

## Checklist

- Set `CSE_PROFILE=production`. It switches to JSON logs written to `<data_dir>/logs/contextsearch.log`, binds to `0.0.0.0` and requires the allow-list below.
- Set `CSE_API_KEY` to a long random value. Without it anyone who can reach the port can search and add folders.
- Set `CSE_ALLOWED_ROOTS` to the folders that may be indexed, for example `/srv/docs,/mnt/share`. In production a folder outside these roots is refused.
- Put the data folder on local disk (SQLite and network file systems do not mix well) with room for the database: roughly the size of the extracted text plus `chunks x dimension x 4 bytes` per model, plus uploads.
- Decide on the models. `CSE_EMBEDDING_MODELS` seeds the dense indexes; the first run downloads them into `HF_HOME` (default `~/.cache/huggingface`). For an air-gapped host, download on a connected machine and copy the cache folder over, or set `HF_HUB_OFFLINE=1` after copying.
- Run `contextsearch doctor` as the service user. It checks SQLite FTS5, torch and the device, the model cache, and that the data folder is writable.
- One process per data folder. Do not run several uvicorn workers or several `serve` commands on the same folder: the vector indexes and the watcher live in memory.
- Front it with a reverse proxy if it needs TLS or a hostname. The app itself speaks plain HTTP.

## Linux with systemd

```bash
sudo useradd --system --create-home --home /var/lib/contextsearch contextsearch
sudo -u contextsearch python3 -m venv /var/lib/contextsearch/venv
sudo -u contextsearch /var/lib/contextsearch/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
sudo -u contextsearch /var/lib/contextsearch/venv/bin/pip install "contextsearch[dense] @ git+https://github.com/inboxpraveen/context-search-engine.git"
```

Skip the first `pip install torch` line on a machine with an NVIDIA GPU and a CUDA driver; the default wheels then use it.

`/etc/contextsearch.env`:

```
CSE_PROFILE=production
CSE_DATA_DIR=/var/lib/contextsearch/data
CSE_API_KEY=change-me-to-something-long
CSE_ALLOWED_ROOTS=/srv/docs
CSE_EMBEDDING_MODELS=BAAI/bge-small-en-v1.5
CSE_HOST=127.0.0.1
CSE_PORT=8080
```

`/etc/systemd/system/contextsearch.service`:

```ini
[Unit]
Description=contextsearch document search
After=network.target

[Service]
User=contextsearch
EnvironmentFile=/etc/contextsearch.env
WorkingDirectory=/var/lib/contextsearch
ExecStart=/var/lib/contextsearch/venv/bin/contextsearch serve
Restart=on-failure
RestartSec=5
# torch and the models want a bit of headroom
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now contextsearch
sudo journalctl -u contextsearch -f
```

`WorkingDirectory` matters when you use a `.env` file instead of `EnvironmentFile`: the file is read from the working directory. Then add the folders:

```bash
curl -X POST http://127.0.0.1:8080/api/v1/sources -H "X-API-Key: change-me-to-something-long" \
     -H "Content-Type: application/json" -d '{"path": "/srv/docs", "name": "Shared documents"}'
```

The service user needs read access to every indexed folder. On a network share mounted with CIFS or NFS, file system events usually do not arrive; the periodic scan (`CSE_SCAN_INTERVAL_SECONDS`, default 300) keeps the index current.

## Windows

The same package runs on Windows 10 and 11 with Python from python.org or conda; `pip install -e ".[dense]"` picks the CPU build of torch by default.

For a service, the simplest reliable route is a scheduled task that runs at startup:

```powershell
$action = New-ScheduledTaskAction -Execute "C:\contextsearch\venv\Scripts\contextsearch.exe" -Argument "serve --profile production" -WorkingDirectory "C:\contextsearch"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName contextsearch -Action $action -Trigger $trigger -User "SYSTEM" -RunLevel Highest
```

Put the settings in `C:\contextsearch\.env` (the working directory of the task). Paths in `CSE_ALLOWED_ROOTS` use backslashes or forward slashes, either works: `CSE_ALLOWED_ROOTS=D:\Shares\Docs,\\fileserver\policies`. Files that are open in Word or Acrobat are skipped with a warning and picked up on the next scan.

## Docker

```bash
docker build -t contextsearch -f docker/Dockerfile .
docker run -d --name contextsearch -p 8080:8080 \
  -v cse-data:/data -v /srv/docs:/docs:ro \
  -e CSE_API_KEY=change-me -e CSE_ALLOWED_ROOTS=/docs \
  contextsearch
```

The image is `python:3.12-slim` with the CPU build of torch, the dense and OCR extras, a non-root user and a health check on `/api/v1/health`. `/data` holds the database, uploads, logs and the model cache (`HF_HOME=/data/models`), so keep it on a volume. Mount the documents read-only and add `/docs` as a source through the UI or the API. `docker/docker-compose.yml` does all of this; copy it, change the key and the mount, and `docker compose up -d`.

To run the test suite inside the image, which is a quick way to confirm the Linux build: `docker run --rm contextsearch python -m pytest -q`.

For a GPU host, build from an NVIDIA CUDA base image, drop the `--index-url` line so pip installs the CUDA wheels, and start with `--gpus all`.

## Reverse proxy

nginx in front for TLS and a hostname. Uploads can be large, and a first search after a restart may take a few seconds while a model loads:

```nginx
server {
    listen 443 ssl;
    server_name search.example.internal;
    client_max_body_size 60m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Request-ID $request_id;
        proxy_read_timeout 300s;
    }
}
```

`X-Request-ID` from the proxy is echoed by the app, so a proxy log line and an app log line can be matched.

## Backups and upgrades

The data folder is the whole state. Back it up with the server stopped, or use `sqlite3 contextsearch.db ".backup /backups/contextsearch-$(date +%F).db"` while it runs (WAL mode allows that). Uploads live next to the database in `uploads/`. Watched folders are not copied; they are re-read from their original location.

Upgrading the package keeps the database: the schema version is stored in the file and the app refuses to open a database written by a newer version. After an upgrade run `contextsearch doctor`, then `contextsearch reindex` if the changelog says the chunking changed (the UI shows a Reindex button with the chunking signature).

`contextsearch db vacuum` reclaims space after many deletions; `contextsearch db optimize` merges the FTS index segments after a large import.

## Sizing

See [PERFORMANCE.md](PERFORMANCE.md) for measured numbers. Short version: a 4-core machine with 8 GB handles tens of thousands of documents comfortably with the default model; embedding throughput on CPU is a few hundred chunks per second for small models, so a first index of 100k pages takes an hour or two, and a GPU brings that down to minutes. Searches are milliseconds for keyword, tens of milliseconds for hybrid, a few hundred with reranking on CPU.

## Moving from version 2

Version 3 is a rewrite. The Flask app on port 5000 with `faiss_index.idx` and the two pickle files is gone, and those files cannot be read by version 3. The install, the API paths and the storage all changed:

| Version 2 | Version 3 |
|---|---|
| `python app.py`, port 5000 | `contextsearch serve`, port 8080 |
| `/search`, `/upload`, `/documents`, `/config` | `/api/v1/search`, `/api/v1/documents/upload`, `/api/v1/documents`, `/api/v1/system/config` |
| `app_config.json` edited from the UI | `CSE_*` environment variables or a `.env` file; search options are per request |
| `uploads/` next to the code | `<data_dir>/uploads/` |
| one model, `dimension` typed by hand | any number of models, dimension read from the model |
| `distilbert-base-uncased` mean pooling | `BAAI/bge-small-en-v1.5` (a retrieval model) plus BM25, fused |

To migrate: install version 3, start it, and either re-upload the files or, better, point it at the folder that holds them. Delete the old `*.idx`, `*.pkl` and `app_config.json` files once you are happy. The version 2 uploads folder is a normal folder and can be added as a source directly.
