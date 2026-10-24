"""Deterministic admission/cancellation tests with tiny uploads and fake workers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import threading
from unittest.mock import Mock

from fastapi import FastAPI
import httpx
from PIL import Image
import pytest

from be.inference_admission import admission_for, inference_lifespan
from be.routes import detection
from be.upload_security import DetectBodyLimitMiddleware


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (12, 8), (1, 2, 3)).save(stream, "PNG")
    return stream.getvalue()


def make_app():
    target = FastAPI(lifespan=inference_lifespan)
    target.add_middleware(DetectBodyLimitMiddleware)
    target.include_router(detection.router)
    admission_for(target)

    @target.get("/test-probe")
    async def probe():
        return {"responsive": True}

    return target


@pytest.fixture
def target():
    application = make_app()
    yield application
    admission_for(application).shutdown()


def client_for(target):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=target, raise_app_exceptions=False),
                             base_url="http://test")


async def post(client, contents=None):
    return await client.post("/detect", files={
        "file": ("test.png", image_bytes() if contents is None else contents, "image/png")})


async def checkpoint(awaitable):
    # Test deadlock detection only; never a production/native inference timeout.
    return await asyncio.wait_for(awaitable, 3)


class BlockedService:
    def __init__(self, loop, fail_first=False):
        self.loop = loop
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Queue()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.calls = self.active = self.peak = 0
        self.threads = set()
        self.fail_first = fail_first

    def detect_image(self, image):
        assert threading.get_ident() != self.loop_thread, "Inference blocked the event loop"
        with self.lock:
            self.calls += 1
            call = self.calls
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.get_ident())
        self.loop.call_soon_threadsafe(self.entered.put_nowait, call)
        try:
            assert self.release.wait(5), "Test did not release its fake worker"
            if call == 1 and self.fail_first:
                raise RuntimeError("INTERNAL_FAKE_WORKER_DETAILS")
            return {"results": []}
        finally:
            with self.lock:
                self.active -= 1


def capture_submissions(monkeypatch, target):
    admission = admission_for(target)
    original = admission.submit
    futures = []
    def submit(*args):
        future = original(*args)
        futures.append(future)
        return future
    monkeypatch.setattr(admission, "submit", submit)
    return futures


def test_t15_busy_before_read_and_only_one_worker_then_third_succeeds(target, monkeypatch):
    async def scenario():
        service = BlockedService(asyncio.get_running_loop())
        monkeypatch.setattr(detection, "_service", service)
        futures = capture_submissions(monkeypatch, target)
        original_read = detection.read_upload
        reads = []
        async def read(file):
            reads.append(file)
            return await original_read(file)
        monkeypatch.setattr(detection, "read_upload", read)
        stages = []
        for name in ["validate_image_header", "decode_image", "validate_decoded_image"]:
            original = getattr(detection, name)
            def stage(*args, function=original):
                stages.append(threading.get_ident())
                return function(*args)
            monkeypatch.setattr(detection, name, stage)
        async with client_for(target) as client:
            first = asyncio.create_task(post(client))
            try:
                await checkpoint(service.entered.get())
                second = await checkpoint(post(client))
                assert second.status_code == 503
                assert second.json() == {"detail": "Inference busy; try again later"}
                assert service.active == service.peak == 1
                assert not first.done() and not service.release.is_set()
                assert len(reads) == len(futures) == 1
                assert len(stages) == 3 and set(stages) == service.threads
                assert admission_for(target)._executor._max_workers == 1
                service.release.set()
                assert (await checkpoint(first)).json() == {"results": []}
                third = await checkpoint(post(client))
                assert third.status_code == 200 and third.json() == {"results": []}
                assert service.calls == 2 and service.peak == 1
                assert len(service.threads) == 1  # The same bounded pool is reused.
            finally:
                service.release.set()
                await asyncio.gather(first, return_exceptions=True)
    asyncio.run(scenario())


def test_t17_loop_serves_other_work_while_inference_is_blocked(target, monkeypatch):
    async def scenario():
        loop = asyncio.get_running_loop()
        service = BlockedService(loop)
        monkeypatch.setattr(detection, "_service", service)
        async with client_for(target) as client:
            first = asyncio.create_task(post(client))
            try:
                await checkpoint(service.entered.get())
                heartbeat = asyncio.Event()
                loop.call_soon(heartbeat.set)
                await checkpoint(heartbeat.wait())
                probe = await checkpoint(client.get("/test-probe"))
                assert probe.json() == {"responsive": True}
                assert service.active == 1 and not first.done()
                assert not service.release.is_set()
            finally:
                service.release.set()
                await checkpoint(first)
    asyncio.run(scenario())


@pytest.mark.parametrize("fail_first", [False, True])
def test_t16_http_cancellation_keeps_running_worker_admitted_until_exit(target, monkeypatch, fail_first):
    async def scenario():
        service = BlockedService(asyncio.get_running_loop(), fail_first=fail_first)
        monkeypatch.setattr(detection, "_service", service)
        futures = capture_submissions(monkeypatch, target)
        async with client_for(target) as client:
            first = asyncio.create_task(post(client))
            try:
                await checkpoint(service.entered.get())
                assert futures[0].running()
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert not futures[0].cancelled() and service.active == 1
                assert (await checkpoint(post(client))).status_code == 503
                assert service.calls == 1 and not service.release.is_set()
                service.release.set()
                if fail_first:
                    with pytest.raises(RuntimeError, match="INTERNAL_FAKE_WORKER_DETAILS"):
                        await checkpoint(asyncio.wrap_future(futures[0]))
                else:
                    await checkpoint(asyncio.wrap_future(futures[0]))
                third = await checkpoint(post(client))
                assert third.status_code == 200 and third.json() == {"results": []}
                assert service.active == 0 and service.peak == 1
            finally:
                service.release.set()
                await asyncio.gather(first, return_exceptions=True)
    asyncio.run(scenario())


def test_t16_cancel_during_file_read_releases_without_submitting_worker(target, monkeypatch):
    async def scenario():
        reading = asyncio.Event()
        never_release = asyncio.Event()
        original = detection.read_upload
        service = Mock()
        service.detect_image.return_value = {"results": []}
        monkeypatch.setattr(detection, "_service", service)
        submissions = capture_submissions(monkeypatch, target)
        async def held_read(file):
            reading.set()
            await never_release.wait()
        monkeypatch.setattr(detection, "read_upload", held_read)
        async with client_for(target) as client:
            first = asyncio.create_task(post(client))
            try:
                await checkpoint(reading.wait())
                assert (await checkpoint(post(client))).status_code == 503
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert submissions == []
                monkeypatch.setattr(detection, "read_upload", original)
                assert (await checkpoint(post(client))).status_code == 200
                service.detect_image.assert_called_once()
            finally:
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
    asyncio.run(scenario())


def test_t16_cancel_submitted_but_not_started_future(target, monkeypatch):
    async def scenario():
        loop = asyncio.get_running_loop()
        occupied = asyncio.Event()
        unblock = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        admission = admission_for(target)
        # Test-only occupation creates a deterministic queued-before-start case.
        admission._executor = pool
        def occupant():
            loop.call_soon_threadsafe(occupied.set)
            assert unblock.wait(5)
        occupant_future = pool.submit(occupant)
        await checkpoint(occupied.wait())
        submitted = asyncio.Event()
        futures = capture_submissions(monkeypatch, target)
        captured_submit = admission.submit
        def submit(*args):
            future = captured_submit(*args)
            submitted.set()
            return future
        monkeypatch.setattr(admission, "submit", submit)
        service = Mock()
        service.detect_image.return_value = {"results": []}
        monkeypatch.setattr(detection, "_service", service)
        async with client_for(target) as client:
            first = asyncio.create_task(post(client))
            try:
                await checkpoint(submitted.wait())
                assert not futures[0].running()
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert futures[0].cancelled()
                service.detect_image.assert_not_called()
                lease = admission.try_acquire()
                assert lease is not None
                lease.release()
                unblock.set()
                await checkpoint(asyncio.wrap_future(occupant_future))
                assert (await checkpoint(post(client))).status_code == 200
                service.detect_image.assert_called_once()
            finally:
                unblock.set()
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("contents,status", [(b"", 400), (b"fake-image", 415),
                                               (b"\x89PNG\r\n\x1a\nbroken", 400)])
def test_validation_failure_before_or_inside_worker_does_not_leak_slot(target, monkeypatch, contents, status):
    async def scenario():
        service = Mock()
        service.detect_image.return_value = {"results": []}
        monkeypatch.setattr(detection, "_service", service)
        async with client_for(target) as client:
            assert (await checkpoint(post(client, contents))).status_code == status
            service.detect_image.assert_not_called()
            assert (await checkpoint(post(client))).status_code == 200
            service.detect_image.assert_called_once()
    asyncio.run(scenario())


def test_parse_failure_does_not_occupy_admission(target, monkeypatch):
    async def scenario():
        service = Mock()
        service.detect_image.return_value = {"results": []}
        monkeypatch.setattr(detection, "_service", service)
        async with client_for(target) as client:
            invalid = await client.post("/detect", content=b"broken", headers={"Content-Type": "multipart/form-data"})
            assert invalid.status_code == 400
            service.detect_image.assert_not_called()
            assert (await checkpoint(post(client))).status_code == 200
    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["worker", "submit"])
def test_failure_does_not_leak_slot_or_expose_worker_details(target, monkeypatch, stage):
    async def scenario():
        marker = "INTERNAL_FAKE_PATH_NOT_A_SECRET"
        service = Mock()
        service.detect_image.return_value = {"results": []}
        admission = admission_for(target)
        original = admission.submit
        if stage == "worker":
            service.detect_image.side_effect = RuntimeError(marker)
        else:
            monkeypatch.setattr(admission, "submit", Mock(side_effect=RuntimeError(marker)))
        monkeypatch.setattr(detection, "_service", service)
        async with client_for(target) as client:
            response = await checkpoint(post(client))
            assert response.status_code == 500
            assert marker not in response.text and "Traceback" not in response.text
            service.detect_image.side_effect = None
            monkeypatch.setattr(admission, "submit", original)
            assert (await checkpoint(post(client))).status_code == 200
    asyncio.run(scenario())


def test_independent_apps_do_not_share_admission_state(target, monkeypatch):
    async def scenario():
        second_app = make_app()
        assert admission_for(target) is not admission_for(second_app)
        service = BlockedService(asyncio.get_running_loop())
        monkeypatch.setattr(detection, "_service", service)
        async with client_for(target) as one, client_for(second_app) as two:
            first = asyncio.create_task(post(one))
            second = None
            try:
                await checkpoint(service.entered.get())
                second = asyncio.create_task(post(two))
                await checkpoint(service.entered.get())
                assert service.active == 2  # One per independent app, not one global lock.
                assert (await checkpoint(post(one))).status_code == 503
                assert (await checkpoint(post(two))).status_code == 503
                service.release.set()
                assert (await checkpoint(first)).status_code == 200
                assert (await checkpoint(second)).status_code == 200
            finally:
                service.release.set()
                await asyncio.gather(*[t for t in [first, second] if t is not None], return_exceptions=True)
                await asyncio.to_thread(admission_for(second_app).shutdown)
    asyncio.run(scenario())


def test_lifespan_shutdown_drains_cancelled_worker_without_blocking_loop(target, monkeypatch):
    async def explicit():
        loop = asyncio.get_running_loop()
        service = BlockedService(loop)
        monkeypatch.setattr(detection, "_service", service)
        context = target.router.lifespan_context(target)
        await context.__aenter__()
        first = None
        exiting = None
        async with client_for(target) as client:
            try:
                first = asyncio.create_task(post(client))
                await checkpoint(service.entered.get())
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                admission = admission_for(target)
                entered = asyncio.Event()
                original = admission._executor.shutdown
                def shutdown(*args, **kwargs):
                    loop.call_soon_threadsafe(entered.set)
                    return original(*args, **kwargs)
                monkeypatch.setattr(admission._executor, "shutdown", shutdown)
                exiting = asyncio.create_task(context.__aexit__(None, None, None))
                await checkpoint(entered.wait())
                assert not exiting.done() and service.active == 1
                assert (await checkpoint(client.get("/test-probe"))).status_code == 200
                assert (await checkpoint(post(client))).status_code == 503
                service.release.set()
                await checkpoint(exiting)
                assert service.active == 0
                assert (await checkpoint(post(client))).status_code == 200
            finally:
                service.release.set()
                if first is not None:
                    await asyncio.gather(first, return_exceptions=True)
                if exiting is not None:
                    await asyncio.gather(exiting, return_exceptions=True)
                else:
                    await context.__aexit__(None, None, None)
    asyncio.run(explicit())


def test_cpu_demo_startup_explicitly_uses_one_worker(monkeypatch):
    import uvicorn
    from scripts import run_backend
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    run_backend.main()
    assert run.call_args.kwargs["workers"] == 1
    docker = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    command = json.loads(next(line[4:] for line in docker.splitlines() if line.startswith("CMD ")))
    assert command[command.index("--workers")+1] == "1"
    assert command[command.index("--port")+1] == "8000"
