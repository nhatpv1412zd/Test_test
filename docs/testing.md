# Lightweight Unit/API tests

Use Python 3.10 and an isolated disposable environment. The project's existing
Windows `.venv` has an unavailable base interpreter and is not a verified runtime;
these commands create a separate environment in the OS temporary directory.
Run setup from the project root. They do not install the production AI stack.

## Linux / WSL

```bash
python3.10 -m venv /tmp/alpr-unit-api-tests
/tmp/alpr-unit-api-tests/bin/python -m pip install -r requirements-test.txt
/tmp/alpr-unit-api-tests/bin/python -B scripts/run_unit_api_tests.py --collect-only
/tmp/alpr-unit-api-tests/bin/python -B scripts/run_unit_api_tests.py
```

The interpreter must support venv/pip; if unavailable, arrange a working Python
environment before setup. Do not silently use the broken project `.venv`. A
previously approved disposable environment can be reused when its versions match.
Top-level versions are pinned in `requirements-test.txt` for Python 3.10; this is
not a full transitive dependency lock. Record `python --version` and
`python -m pip freeze` for final reproducibility evidence. No environment creation
or install is performed by the runner itself.

## Windows PowerShell

With a separately installed working Python 3.10:

```powershell
$testEnvPath = Join-Path ([IO.Path]::GetTempPath()) 'alpr-unit-api-tests'
py -3.10 -m venv $testEnvPath
$testPython = Join-Path $testEnvPath 'Scripts/python.exe'
& $testPython -m pip install -r requirements-test.txt
& $testPython -B scripts/run_unit_api_tests.py --collect-only
& $testPython -B scripts/run_unit_api_tests.py
```

These are setup instructions, not evidence that Windows execution has passed.
Do not repair/replace the project's `.venv` as part of these test commands.

## Collection policy and runner

`pytest.ini` is the single routine collection policy. Its `testpaths` names:

- `tests/test_api.py`
- `tests/test_upload_security.py`
- `tests/test_inference_admission.py`
- `tests/test_service.py`
- `tests/test_preprocessing.py`
- `tests/test_postprocessing.py`
- `tests/test_paths.py`

New wrapper/config cases live in the existing service/paths modules. `addopts`
ignores `experiments/` and `tests/test_inference_smoke.py` before collection;
`norecursedirs` also excludes archives/vendor directories. Root-level
`python -B -m pytest --collect-only -q` uses this scope. Do not override ini/options
or add historical paths for routine verification.

The preferred runner resolves the root from its own file, changes to that root
for pytest, and restores the caller's working directory afterward. For example,
from another directory use `python -B <project>/scripts/run_unit_api_tests.py`.
It supports only `--collect-only` and optional `-k EXPRESSION`; it does not accept
arbitrary file paths that could expand the routine suite. Target the additions:

```bash
python -B scripts/run_unit_api_tests.py -k 'yolo_wrapper or test_config_'
```

Both collection and execution install a stdlib import guard before pytest/test
imports. It rejects `ultralytics`, `easyocr`, `torch`, `torchvision`, records even
caught forbidden import attempts, checks already-loaded modules, and returns
nonzero on violation. Required lightweight dependencies are checked before
collection; test-module import failures remain visible. A nonzero pytest code
is preserved, including no tests selected or collection failures. The runner
disables plugin autoload, ignores inherited PYTEST_ADDOPTS/PYTEST_PLUGINS, and
disables bytecode and pytest cache generation. Direct pytest uses safe collection
and conftest isolation, but does not provide the runner's preflight/import guard.

Only app-specific environment keys are isolated and restored. Test values match
the documented CPU defaults; `PYTHON_DOTENV_DISABLED=1` prevents private `.env`
loading with the pinned python-dotenv. `tests/conftest.py` applies them before
test-module imports, including direct routine pytest. Function-scoped headroom
mocking covers fake Reader/service/API tests; dedicated refusal cases override it
and retain their assertions. Config regressions use fresh subprocesses, do not
read model weights, and assert no dotenv parser/file access. None of this changes
production config or the Windows memory policy.

The YOLO lazy-factory regression intercepts only that test's Python import-call
boundary and returns a fake factory, without installing a model module in
sys.modules or disabling the guard. Other wrapper cases inject fake model results.
Raw-ASGI upload helpers explicitly drain app admissions and assert pool threads
have exited. R02 fake worker synchronization/cancellation tests remain unchanged.

The lightweight suite needs repository source, Dockerfile text and the two small
dataset YAML configs, not checkpoints, GPU, image datasets or external services.
One existing preprocessing regression may skip if its optional saved diagnostic
image is absent. Missing dependencies must not be counted as successful skips.
R01/R02's historical result is 164 passed; accepted T2-04-C later collected and
passed 175 cases (see Verification scope below). Record actual future
collected/passed/skipped counts rather than treating a historical count as fixed.

## Optional real-model smoke (outside routine verification)

Only after explicit authorization, working model dependencies, weights/sample
and OCR assets are available, use a separate process without the guarded runner:

```bash
python -B -m pytest -o addopts='' -q tests/test_inference_smoke.py
```

This overrides the smoke exclusion deliberately. Conftest does not apply fake
headroom or routine environment overrides to this explicit smoke command.
Real packages/assets may download or need native runtime resources. The smoke
checks pipeline execution, not OCR accuracy across a dataset. It is not executed
in T2-04-B. Historical experiments OPT-01/04/06 and all archived tests/tools remain
outside routine verification and must not be rerun as substitutes for Unit/API.

## Verification scope

Owner accepted [T2-04-C final evidence](../experiments/tier2/t2-04-c-final-verification-report.md):
**175 collected, 175 passed, 0 failed, 0 skipped** in **WSL Python 3.10.12,
lightweight/source/mock** scope. The **11 targeted tests are included in the
175-case suite**; **14 controlled runner/guard checks were performed separately**.
These are previously accepted results, not new executions during T2-07-B.

Windows-native, Docker runtime, real-model smoke/accuracy, remote CI and fresh
setup on a second machine are outside this accepted evidence. CI workflow
alignment and remote results belong to T2-05. Per-app
capacity-one is not a global limit, cancellation cannot terminate native threads,
and mocks do not establish total peak RAM bounds or handling of every native crash.

See [local setup/API usage](../README.md#setup), [API reference](../README.md#api)
and [Docker readiness/status](docker-deployment.md) for the separate runtime paths.
