"""Run the pytest.ini lightweight suite from any working directory."""

import sys

sys.dont_write_bytecode = True

import argparse
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.unit_api_test_support import ModelImportGuard, isolated_test_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("-k", metavar="EXPRESSION", help="Select targeted cases within the routine suite")
    options = parser.parse_args(argv)
    guard = ModelImportGuard()
    sys.meta_path.insert(0, guard)
    previous_cwd = Path.cwd()
    controls = ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    previous_controls = {name: os.environ.get(name) for name in controls}
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    code = 2
    try:
        os.chdir(ROOT)
        with isolated_test_environment():
            if guard.loaded_modules():
                print("Refusing an environment with preloaded model modules", file=sys.stderr)
                code = 1
            else:
                required = ("pytest", "fastapi", "uvicorn", "multipart", "numpy", "cv2", "PIL", "httpx", "dotenv")
                missing = [name for name in required if importlib.util.find_spec(name) is None]
                if missing:
                    print("Missing required test dependencies: " + ", ".join(missing), file=sys.stderr)
                else:
                    import pytest

                    args = ["-c", str(ROOT / "pytest.ini"), "-q"]
                    if options.collect_only:
                        args.append("--collect-only")
                    if options.k:
                        args.extend(["-k", options.k])
                    code = int(pytest.main(args))
    finally:
        attempts = guard.attempts
        loaded = guard.loaded_modules()
        print("Forbidden model import attempts:", attempts)
        print("Loaded forbidden modules:", loaded)
        sys.meta_path.remove(guard)
        os.chdir(previous_cwd)
        for name, value in previous_controls.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return code or int(bool(attempts or loaded))


if __name__ == "__main__":
    raise SystemExit(main())
