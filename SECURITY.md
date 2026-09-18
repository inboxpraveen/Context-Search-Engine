# Security

contextsearch is meant to run on an internal network, on a machine you control. This page says what the software does to protect the index and what you have to do yourself.

## What the software does

- **API key.** With `CSE_API_KEY` set, every `/api` request must present the key as `X-API-Key` or `Authorization: Bearer`. Comparison is constant time. `GET /api/v1/health` answers without a key with only the status and version. The key can also be given as `?api_key=` for links a browser opens directly (the file endpoint); that form ends up in browser history and proxy logs, so prefer the header from scripts.
- **Folder allow-list.** In the production profile a folder can only be added as a source when it sits inside `CSE_ALLOWED_ROOTS`. The development profile allows any path, which is convenient on a laptop and wrong on a server.
- **Files by id only.** The file and page endpoints locate content by document id; no endpoint accepts a client-supplied path.
- **Uploads.** File names are sanitised, types are limited to the configured extensions, size is limited by `CSE_MAX_UPLOAD_MB`, and files are stored under the data folder, never executed.
- **Responses.** Errors never include stack traces; they carry a request id that maps to the server log. Settings marked secret are masked in `/api/v1/system/config`. Responses set `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN` and `Referrer-Policy: same-origin`.
- **UI.** All text from documents is escaped before rendering; highlights are applied by offsets, not by injecting markup from the server. The UI loads no external scripts, styles or fonts.
- **Models.** Model weights come from the Hugging Face hub in safetensors format where available. `trust_remote_code` is enabled only for the nomic presets that need it; other models never execute code from the hub.
- **Nothing leaves the machine.** No telemetry, no calls to any service other than the model download you configure.

## What you have to do

- Set `CSE_API_KEY` to a long random value and `CSE_PROFILE=production` on any machine other people can reach.
- Set `CSE_ALLOWED_ROOTS`. The index contains the full text of everything under the sources; anyone with the key can read all of it through search and the page endpoint.
- Put TLS in front with a reverse proxy if the network is not trusted, and restrict who can reach the port.
- Run the service as a user that can read the documents and nothing more, and keep the data folder readable only by that user; it holds the full text and the uploads.
- Keep the process count at one per data folder. Two writers on one SQLite file corrupt nothing but do fight over locks.
- Treat the query log and events as data about your users; `CSE_QUERY_LOG=false` turns the log off.

## Reporting a vulnerability

Email the address in the repository profile with a description and steps to reproduce, and allow a reasonable time for a fix before publishing. Please do not open a public issue for something that could put other people's installations at risk.
