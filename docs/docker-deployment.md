# Docker CPU deployment and T2-02-C checklist

Configuration prepared in T2-02-B; **all build/runtime gates below are NOT VERIFIED**.
T2-02-A/B are approved in audit/implementation/static scope. **T2-02 remains
PARTIAL / RUNTIME VERIFICATION BLOCKED**: [approved C1 readiness evidence](../experiments/tier2/t2-02-c1-environment-readiness-report.md)
is **BLOCKED** on the development laptop, and **C2 V01–V10 are NOT EXECUTED**.
Do not execute Docker workloads on that laptop. C2 requires a separately designated,
explicitly authorized suitable machine that first passes C1 on that same target.
One-command runtime reproducibility remains **DEFERRED_VERIFICATION (DOC-03-02)**.

C1 later parsed **8/8 documentation Python snippets** using WSL Python 3.10.12,
closing B's optional AST-check gap. This is syntax-only evidence, not execution
of those snippets, native imports or model inference. Prior reports remain historical.
No Docker, native-model or accuracy success is claimed by these instructions.

## Profile and prerequisites

- One backend service, Linux/amd64, Python 3.10 (`python:3.10-slim-bookworm`), CPU
  Torch 2.14.1+cpu / TorchVision 0.29.1+cpu. ARM/emulation and Windows native are
  outside this selected profile; Docker Desktop must have adequate VM resources.
- Working Docker Engine/Compose, network to Docker Hub, Debian repositories,
  PyPI and the official PyTorch CPU wheel host. First OCR also needs access to
  EasyOCR release downloads. Bash, curl and standard Linux coreutils are used by the C examples;
  Windows users can use WSL against an approved Docker-capable environment.
- From the project root retain Dockerfile, docker-compose.yml, .dockerignore,
  requirements-docker.txt, be/, src/, .env.example, weights/best.pt and
  assets/samples/. Do not transfer private .env, virtualenvs, training datasets or
  historical experiment outputs just to build this API profile.
- Approved checkpoint SHA-256:
  `54d492d77715ce0eca1ad3bd9195a30541e903c661fdc178731d30bf53bdc35f`.
  Dockerfile copies it to `/app/weights/best.pt` at build time; MODEL_PATH remains weights/best.pt,
  resolved against PROJECT_ROOT `/app`. No host bind can shadow it in this Compose.
- Dockerfile copies samples to `/app/assets/samples/`; they remain on the host for curl.
  No runtime host-side weights/data/assets directory or private OCR cache mount
  is needed after the image is built. The source checkpoint/samples are needed
  on the build machine, including when transferred outside Git.
- Port 8000 must be available. Record machine OS/architecture, RAM/disk and
  Docker/VM allocation. There is no measured minimum RAM or total-peak guarantee.
  Do not test by deliberately provoking OOM or running concurrent requests.

Examples assume the verification machine can reach the published Docker port on
127.0.0.1. For a remote Engine, perform the HTTP checks on that server or explicitly
use its reachable address; a local client's loopback is not the remote host.

The default backend deployment is one command:

```bash
docker compose up --build -d
```

Status/logs/liveness, after approval to run C:

```bash
docker compose ps
docker compose logs --no-color --timestamps backend
curl -i http://127.0.0.1:8000/openapi.json
```

OpenAPI/docs prove HTTP liveness only. Models initialize lazily. YOLO failure
can return `results: []`; OCR failures can retain detections with empty text.
Neither an empty response nor HTTP 200 alone passes the inference gate.

## Dependency policy

Docker installs only requirements-docker.txt, not requirements.txt or the
lightweight requirements-test.txt. Local/test startup instructions stay separate.
Do not run a later `pip install -r requirements.txt` inside this image: it would
reintroduce the desktop OpenCV dependency graph.

The dedicated manifest uses hashed direct URLs from the official CPU wheel index
for CPython 3.10/Linux x86_64. The resolver is given those exact Torch candidates;
it must fail on an incompatible dependency instead of choosing a CUDA build.
CPU TorchVision metadata requires Torch >=2.14.0, which includes 2.14.1+cpu.
This is metadata evidence, not proof that native imports/checkpoint loading work.

Ultralytics uses the official ultralytics-opencv-headless distribution at the
same public version 8.4.177 inspected in A. Its non-extra dependency metadata
differs from standard Ultralytics only by the headless OpenCV requirement; it
still provides `from ultralytics import YOLO`. EasyOCR stays at 1.7.2. Both select
opencv-python-headless 5.0.0.93; no desktop/contrib variant is deliberately installed.
API/config pins come from the approved lightweight stack; the full AI runtime
was not tested by that suite. No application algorithm or inference setting changes.

`--only-binary=:all:` avoids an unexpected source compilation on a small machine.
No matching wheel is a build failure to record, not permission to change versions
or compile heavy packages silently. Top-level pins and two wheel hashes are not
a complete transitive lock or immutable base-image digest. C must record actual
base/image ID, resolved versions, pip check and native/model results. The
Ultralytics dependency `nvidia-ml-py` is a monitoring binding, not a CUDA Torch build;
do not reject it merely because its name contains nvidia.

Sources for the policy: [official CPU indexes](https://download.pytorch.org/whl/cpu/),
[headless package release](https://pypi.org/project/ultralytics-opencv-headless/8.4.177/)
and [EasyOCR Reader parameters](https://www.jaided.ai/easyocr/documentation/).
Actual combined compatibility remains **DEFERRED_VERIFICATION**.

## Connected first OCR call and persistent cache

Docker sets EASYOCR_MODULE_PATH=/opt/easyocr. EasyOCR uses
`/opt/easyocr/model` and `/opt/easyocr/user_network`; image setup creates the
directories. Compose auto-creates a read/write named volume, easyocr-cache,
mounted at that root. The current image retains its default root user; no
optional non-root/read-only hardening is introduced. Actual permissions/mount
availability must pass V06 on the target machine.

No OCR weights are downloaded at build time or bundled from a private host cache.
With the existing English/CPU Reader, the first request that reaches OCR may
download craft_mlt_25k.pth and english_g2.pth. A request with no detections may
never call OCR, so it does not initialize or verify the OCR resources. Allow the
first download to complete, inspect logs, and record failures if network access
is unavailable. No manual model copy or separate warmup process is required.

The named volume persists across container recreation and `docker compose down`
without `-v`, under the same Compose project identity. A different project name
may select a different volume. Do not run `down -v`, volume prune or delete this
cache as part of ordinary stop/restart. Docker Engine removal/manual volume
deletion also removes persistence. This connected profile does not promise fully
offline operation, even after one successful request.

Frontend deployment stays separate: serve fe/index.html with a static server on
the browser machine. Its fetch URL is http://127.0.0.1:8000/detect; change the
public URL if the backend is remote. The browser's Tailwind CDN is another network
dependency. No additional frontend/proxy service was added.

## Evidence setup for C only

Use a fresh Bash verification session from the project root on the suitable
machine. Commands below are proposals, not evidence of execution in B.
Create an evidence directory outside the project; keep private credentials and
whole environment dumps out of logs. `capture` preserves each command's actual
exit code and writes stdout/stderr separately. Stop at any unexpected nonzero;
do not repair/retry without recording and classifying the failure.

```bash
set +e  # Record nonzero status before any explicit stop; do not rely on errexit.
alpr_evidence_dir=$(mktemp -d /tmp/alpr-t202-c.XXXXXX) || exit $?
capture() {
    local label=$1
    shift
    printf '%q ' "$@" > "$alpr_evidence_dir/$label.command.txt"
    printf '\n' >> "$alpr_evidence_dir/$label.command.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$alpr_evidence_dir/$label.started-utc.txt"
    "$@" > "$alpr_evidence_dir/$label.stdout.txt" 2> "$alpr_evidence_dir/$label.stderr.txt"
    local status=$?
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$alpr_evidence_dir/$label.finished-utc.txt"
    printf '%s\n' "$status" > "$alpr_evidence_dir/$label.exit.txt"
    cat "$alpr_evidence_dir/$label.stdout.txt"
    cat "$alpr_evidence_dir/$label.stderr.txt" >&2
    return "$status"
}
capture metadata-cli docker --version || exit $?
capture metadata-compose docker compose version || exit $?
capture metadata-config docker compose config --quiet || exit $?
```

Record machine resources manually or with appropriate host tooling. Confirm
Docker/VM resources, not just host physical RAM; avoid exposing account/proxy
credentials in metadata. After an HTTP timeout, the native inference thread may
still be running: do not launch another sample/warmup concurrently.

## C gates — every row currently NOT VERIFIED

| Gate | Commands below | Expected result / evidence | Failure interpretation | Status |
|---|---|---|---|---|
| V01 One-command build/start | V01 | Exit 0; build/start stdout/stderr, context transfer size, image ID and timings | Registry/network, resolver, wheel/native install or resources; no silent version change | NOT VERIFIED |
| V02 Running/stable | V02 | One backend, running without crash loop; selected state/restart/OOM fields and logs | Process/config/dependency/resource failure; running does not prove models | NOT VERIFIED |
| V03 HTTP liveness | V03 | HTTP 200 OpenAPI schema includes /detect | Network/port/app startup; no dedicated health endpoint was added | NOT VERIFIED |
| V04 Dependencies/native imports | V04 | pip check exit 0; expected versions, CPU Torch, one OpenCV variant, native imports succeed | Resolver/ABI/shared-library failure; checkpoint still untested | NOT VERIFIED |
| V05 Checkpoint/resources | V05 | Exact approved hash/path; samples present; private .env absent | Packaging/checkpoint mismatch; stop without substituting weights | NOT VERIFIED |
| V06 OCR first-run/cache | V06 before and after V07 | Correct named RW mount, writable dirs; after OCR two nonempty weight files; persistence across recreation | Mount/permissions/download/cache failure; no private host-cache fallback | NOT VERIFIED |
| V07 Positive sample / YOLO+OCR | V07 | Valid JSON with nonempty detection/plate, completed YOLO/OCR logs; no runtime-error interpretation hidden by 200 | Failed initialization/inference or inconclusive OCR, not successful readiness | NOT VERIFIED |
| V08 Invalid upload | V08 | Tiny fake image rejected with HTTP 415 and safe detail, no inference | Validation regression or request sent while worker busy; preserve evidence | NOT VERIFIED |
| V09 One worker/CPU | V09 | CMD workers=1, no reload/replicas/GPU profile; one Uvicorn server process, CPU native result from V04 | Deployment override/profile mismatch; capacity is per app/process | NOT VERIFIED |
| V10 Shutdown/evidence | V10 | Save logs/state before down, preserve named cache and all evidence; clean stop | In-flight/native hang, forced stop or other lifecycle failure; record exit/OOM state | NOT VERIFIED |

### V01

```bash
capture 01-build-start docker compose up --build -d || exit $?
capture 01-image docker image inspect --format '{{.Id}} {{.Os}} {{.Architecture}}' ccnlthd-lpr:latest || exit $?
```

One invocation builds and starts; do not repeat a build just for a screenshot.

### V02

```bash
capture 02-services docker compose ps || exit $?
alpr_container_id=$(docker compose ps -q backend)
capture 02-state docker inspect --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.RestartCount}}' "$alpr_container_id" || exit $?
capture 02-startup-logs docker compose logs --no-color --timestamps backend || exit $?
```

Inspect running/stability and errors, not just command exit. Preserve unexpected
restart/OOM fields. Never dump the full docker inspect environment into evidence.

### V03

```bash
capture 03-http curl -sS --max-time 30 -D "$alpr_evidence_dir/openapi.headers" -o "$alpr_evidence_dir/openapi.json" -w '%{http_code}\n' http://127.0.0.1:8000/openapi.json || exit $?
capture 03-status test "$(cat "$alpr_evidence_dir/03-http.stdout.txt")" = 200 || exit $?
capture 03-schema docker compose exec -T backend python -c 'import json,sys; schema=json.load(sys.stdin); assert "post" in schema["paths"]["/detect"]; print("OpenAPI contains POST /detect")' < "$alpr_evidence_dir/openapi.json" || exit $?
```

Keep response/schema. HTTP liveness remains distinct from V07 model processing.

### V04

This separate short-lived process imports dependencies but **never constructs
YOLO or EasyOCR Reader**. Run it before V07; it exits before model initialization.

```bash
capture 04-pip-check docker compose exec -T backend python -m pip check || exit $?
capture 04-native docker compose exec -T backend python -c 'import importlib.metadata as m; import cv2, numpy, torch, torchvision, easyocr, ultralytics; assert torch.__version__ == "2.14.1+cpu"; assert torchvision.__version__ == "0.29.1+cpu"; assert torch.version.cuda is None and not torch.cuda.is_available(); names={d.metadata["Name"].lower().replace("_", "-") for d in m.distributions()}; assert "opencv-python-headless" in names; assert not names.intersection({"opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"}); print("native imports and CPU profile OK"); print("\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))' || exit $?
```

This is native/dependency evidence, not model or OCR accuracy evidence.

### V05

```bash
capture 05-files docker compose exec -T backend python -c 'from pathlib import Path; import hashlib; from src.utils.config import MODEL_PATH; expected="54d492d77715ce0eca1ad3bd9195a30541e903c661fdc178731d30bf53bdc35f"; p=Path(MODEL_PATH); assert str(p)=="/app/weights/best.pt"; observed=hashlib.sha256(p.read_bytes()).hexdigest(); assert observed==expected; assert Path("/app/assets/samples/z7659940505816_1c2add17a54ecb849b85fbb059965480.jpg").is_file(); assert Path("/app/.env.example").is_file(); assert not Path("/app/.env").exists(); print(str(p), observed)' || exit $?
```

No torch.load or model construction is needed to check file integrity.

### V06 — mount/writability first; weights/persistence after V07

```bash
capture 06-mounts docker inspect --format '{{json .Mounts}}' "$alpr_container_id" || exit $?
capture 06-writable docker compose exec -T backend python -c 'import os, tempfile; from pathlib import Path; assert os.environ.get("EASYOCR_MODULE_PATH")=="/opt/easyocr"; assert Path("/opt/easyocr/model").is_dir(); assert Path("/opt/easyocr/user_network").is_dir(); f=tempfile.TemporaryFile(dir="/opt/easyocr"); f.close(); print("cache directories writable")' || exit $?
```

Check Mounts says named volume, destination /opt/easyocr, RW=true, not a host bind.
Do not require OCR weights in an empty cache before V07; lazy connected download
is the chosen policy. No preloaded home cache is needed.

### V07

The existing image below was visually inspected during B and contains a clear
plate. Its actual Docker YOLO/OCR result is **NOT VERIFIED**. Use exactly this
real path, not a placeholder or a new experimental dataset.

```bash
capture 07-valid curl -sS --max-time 300 -D "$alpr_evidence_dir/valid.headers" -o "$alpr_evidence_dir/valid.json" -w '%{http_code}\n' -F 'file=@assets/samples/z7659940505816_1c2add17a54ecb849b85fbb059965480.jpg;type=image/jpeg' http://127.0.0.1:8000/detect || exit $?
capture 07-status test "$(cat "$alpr_evidence_dir/07-valid.stdout.txt")" = 200 || exit $?
capture 07-response docker compose exec -T backend python -c 'import json,sys; data=json.load(sys.stdin); assert set(data)=={"results"}; results=data["results"]; assert isinstance(results,list) and results; assert all(set(r)=={"confidence","plate_number"} and isinstance(r["confidence"],(int,float)) and isinstance(r["plate_number"],str) for r in results); assert any(r["plate_number"].strip() for r in results); print("approved schema and nonempty recognized plate")' < "$alpr_evidence_dir/valid.json" || exit $?
capture 07-inference-logs docker compose logs --no-color --timestamps backend || exit $?
```

Inspect valid.json: results must be a nonempty list with the approved
confidence/plate_number fields and a nonempty recognized plate. Correlate logs
showing YOLO detections and completed OCR (`status=ok`), without YOLO detect failed
or ocr_runtime_error for this call. An empty results/text response, failed OCR
download or timeout is **not** PASS even when the API preserves HTTP 200. Do not
change thresholds/checkpoint, invent ground truth or rerun accuracy experiments
to turn this gate green. The curl deadline is a client deadline, not a native
thread kill or RAM bound. Review any failure before trying another request.

### V08

Run only when V07 finished and its worker exited. A busy 503 is not a format test.

```bash
printf 'not-an-image' > "$alpr_evidence_dir/invalid.txt"
capture 08-invalid curl -sS --max-time 30 -D "$alpr_evidence_dir/invalid.headers" -o "$alpr_evidence_dir/invalid.json" -w '%{http_code}\n' -F "file=@$alpr_evidence_dir/invalid.txt;filename=invalid.jpg;type=image/jpeg" http://127.0.0.1:8000/detect || exit $?
capture 08-status test "$(cat "$alpr_evidence_dir/08-invalid.stdout.txt")" = 415 || exit $?
capture 08-response docker compose exec -T backend python -c 'import json,sys; data=json.load(sys.stdin); assert data=={"detail":"Unsupported image format"}; print("safe unsupported-format detail")' < "$alpr_evidence_dir/invalid.json" || exit $?
```

Check invalid.json is safe detail JSON. Curl deliberately has no --fail here:
transport exit 0 and HTTP 415 are separate expected observations, both recorded.

### V09

```bash
capture 09-command docker inspect --format '{{json .Config.Cmd}}' "$alpr_container_id" || exit $?
capture 09-processes docker compose top backend || exit $?
```

Require --workers 1, no reload/additional replicas and one Uvicorn server process.
V04 also established the CPU native profile; no GPU device is requested in Compose.
Library internal threads do not equal additional Uvicorn workers.

### V06 continuation — after successful V07/V08/V09

```bash
capture 06-cache-before docker compose exec -T backend python -c '
from pathlib import Path
import hashlib
for name in ("craft_mlt_25k.pth", "english_g2.pth"):
    p = Path("/opt/easyocr/model") / name
    assert p.is_file() and p.stat().st_size > 0
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    print(p.name, h.hexdigest())
' || exit $?
capture 06-recreate docker compose up -d --no-build --force-recreate backend || exit $?
alpr_container_id=$(docker compose ps -q backend)
capture 06-cache-after docker compose exec -T backend python -c '
from pathlib import Path
import hashlib
for name in ("craft_mlt_25k.pth", "english_g2.pth"):
    p = Path("/opt/easyocr/model") / name
    assert p.is_file() and p.stat().st_size > 0
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    print(p.name, h.hexdigest())
' || exit $?
capture 06-persistence cmp "$alpr_evidence_dir/06-cache-before.stdout.txt" "$alpr_evidence_dir/06-cache-after.stdout.txt" || exit $?
```

This recreates sequentially after requests have finished. It starts no second
model/warmup process and submits no extra inference. Cache hashes must persist.
The original V07 logs are already saved before recreation; do not overwrite them.

### V10

```bash
capture 10-final-logs docker compose logs --no-color --timestamps backend || exit $?
capture 10-final-state docker inspect --format '{{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' "$alpr_container_id" || exit $?
capture 10-down docker compose down --timeout 120 || exit $?
capture 10-after docker compose ps -a || exit $?
```

Check deployment containers are stopped/removed and preserve the evidence directory,
machine metadata, source/checkpoint hashes, image IDs, resolved versions and any
failure records. Do not use down -v/prune/delete images or model caches. Normal
down preserves the named volume. Do not interpret capacity-one as global across
processes, cancellation as native-thread termination, or these gates as proof of
total peak RAM or handling every native crash. All results still need Owner Review.
