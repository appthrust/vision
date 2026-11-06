# Vision

Vision is AppThrust's GPU-backed OCR and image-understanding service. Its Go
server supervises vLLM and proxies OpenAI-compatible chat completions to the
baked `Qwen/Qwen3-VL-4B-Instruct-FP8` model. It releases independently of the
platform; there is no web frontend or database.

## Develop

Use Go 1.26 or newer and Bun 1.4.1 for the publisher tests. The public Go SDK is
[`github.com/appthrust/appserver-go`](https://github.com/appthrust/appserver-go)
`v0.1.0`, required normally without a platform checkout or module replacement.

```sh
go build ./...
go test ./...
bun test hack/
docker build --platform linux/amd64 --target server-builder .
```

The builder-only command does not pull vLLM or download model weights. Go tests
exercise HTTP and child-process behavior without a GPU or installed vLLM. A full
image build downloads the pinned model revision, and runtime requires an NVIDIA
GPU. The model has not yet been GPU-tested.

The image starts `/vision-server` on port 8000, with vLLM listening only on
`127.0.0.1:8001`. Run the Go entrypoint with `go run ./cmd/server --listen :8000`.
It exposes:

- `POST /v1/chat/completions`: JSON or streaming responses; model must be empty,
  omitted, or `qwen3-vl-4b-instruct-fp8`. Maximum body size is 8 MiB, with four
  concurrent requests and a 120-second timeout. vLLM permits one image per prompt.
- `GET /v1/models` and `GET /metrics`: proxied to vLLM.
- `GET /healthz`: process liveness; `GET /health`: readiness only after vLLM is
  healthy and lists the baked model.
- `GET /releasez`: SDK release identity; `GET /`: the served model name.

Management routes are not exposed; client credentials and query parameters are
not forwarded upstream. Model weights are baked into the image and runtime is
forced offline. The supervisor restarts failed vLLM processes and reaps their
process groups on shutdown.

`VISION_VLLM_BIN` defaults to `vllm`. For an external test upstream, explicitly
set `VISION_VLLM_BIN=` and `VISION_UPSTREAM` to an HTTP(S) origin without
credentials, path, query, or fragment. `PLATFORM_PROJECT_ID` is used for logging.

## Release

After configuring publication credentials, push a SemVer tag starting at
`v0.2.0`; the compatibility floor already owns `0.1.0`:

```sh
git tag v0.2.0
git push origin v0.2.0
```

`.github/workflows/publish.yml` builds amd64 and arm64 on native Blacksmith
runners, because the model-bake step executes target-architecture code. Each job
pushes `ghcr.io/appthrust/vision:<version>-<arch>` and uploads its digest. The
publisher assembles `ghcr.io/appthrust/vision:<version>` from those immutable
digests, resolves the multi-architecture index with ORAS, and keyless-signs it
with Cosign. The GHCR package must be public after its first push.

The workflow needs `packages:write` and `id-token:write`, repository secret
`CLOUD_CATALOG_ADMIN_TOKEN`, and optional variable `CLOUD_API_URL` (default
`https://api.appthrust.com`). Grant access to organization secrets
`APPTHRUST_GITHUB_ACTIONS_APP_ID` and
`APPTHRUST_GITHUB_ACTIONS_APP_PRIVATE_KEY`; that GitHub App needs Contents write
permission on `appthrust/cloud` to dispatch catalog publication.

The copied publisher has two phases:

```sh
bun hack/publish-entry.ts submit --version 0.2.0 --digest sha256:<image-index-digest>
# Only when submit returns state=snapshot, dispatch using a Cloud installation token:
gh api -X POST repos/appthrust/cloud/dispatches -f event_type=catalog-publish -f 'client_payload[snapshot_id]=<snapshot-id>'
bun hack/publish-entry.ts await --version 0.2.0 --digest sha256:<image-index-digest>
```

`submit` merges persisted version history with `catalog/entry.yaml`, PUTs the
Vision entry, then requests publication. It accepts `queued` (202) or `snapshot`
(200); the workflow dispatches only the latter. `await` needs ORAS 1.3.0 and
Cosign 3.1.3 but no admin token. For at most ten minutes it checks the published
catalog for the exact version and image digest, then verifies the catalog's
signature against the Cloud catalog workflow identity. Dispatch alone is not
publication proof.

## Catalog and image contract

`catalog/entry.yaml` retains the Cloud seed's original `0.1.0` image as its
compatibility floor. New versions use `ghcr.io/appthrust/vision@sha256:...`,
`stream: <major.minor>`, `served: true`, and the only `default: true`; earlier
versions remain served. Version images are immutable: the same version/digest
is idempotent, but replacement is rejected. Workflow concurrency serializes the
read/merge/PUT sequence.

Existing installations moving from the seed image need an explicit version
update with `replaceRepository: true`. Afterwards `follow` follows its stream,
while `pinned` requires an explicit update. `/releasez` reports catalog-rendered
`VISION_RELEASE_VERSION` and `VISION_ARTIFACT_DIGEST` (with optional
`VISION_SOURCE_REVISION` or fallback `VISION_RELEASE_FILE`).
