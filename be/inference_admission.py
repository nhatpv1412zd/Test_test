"""Application-owned, capacity-one admission for synchronous inference jobs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import threading


class _Lease:
    """One release even when cancellation races completion."""

    def __init__(self, slot):
        self._slot = slot
        self._lock = threading.Lock()
        self._released = False

    def release(self):
        with self._lock:
            if not self._released:
                self._released = True
                self._slot.release()


class InferenceAdmission:
    """No waiting admission queue; at most one submitted/running job per app.

    The private executor starts lazily and has exactly one worker. The lease
    covers async file reading too, before work is submitted to that executor.
    """

    def __init__(self):
        self._slot = threading.BoundedSemaphore(1)
        self._state_lock = threading.Lock()
        self._executor = None
        self._stopping = False

    def try_acquire(self):
        with self._state_lock:
            if self._stopping or not self._slot.acquire(blocking=False):
                return None
        return _Lease(self._slot)

    def submit(self, lease, function, *args):
        def run():
            try:
                return function(*args)
            finally:
                # Actual synchronous work has returned or raised. Releasing in
                # the requesting coroutine's finally would be unsafe on cancel.
                lease.release()

        with self._state_lock:
            if self._stopping:
                raise RuntimeError("Inference executor is stopping")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alpr-inference")
            future = self._executor.submit(run)

        def cancelled_before_start(completed):
            # Future.cancel succeeds only if work has not started. Otherwise
            # run() owns release, including if the HTTP requester has gone away.
            if completed.cancelled():
                lease.release()

        future.add_done_callback(cancelled_before_start)
        return future

    def shutdown(self):
        """Drain real work before closing the pool; safe for lifespan restart."""
        with self._state_lock:
            self._stopping = True
            executor = self._executor
        try:
            if executor is not None:
                executor.shutdown(wait=True)
        finally:
            with self._state_lock:
                self._executor = None
                self._stopping = False


def admission_for(app):
    """Each FastAPI instance owns its component, including minimal test apps.

    No await occurs during initialization. dict.setdefault also prevents two
    first callers on different threads from retaining different components.
    Constructing a candidate creates no executor or worker thread.
    """
    state = app.__dict__
    if "inference_admission" not in state:
        state.setdefault("inference_admission", InferenceAdmission())
    return state["inference_admission"]


async def wait_for_worker(future):
    """Cancel queued work if possible, never confuse cancellation with stopping."""
    wrapped = asyncio.wrap_future(future)

    def consume_abandoned_error(completed):
        # A cancelled HTTP request may never await the result/exception again.
        if not completed.cancelled():
            completed.exception()

    wrapped.add_done_callback(consume_abandoned_error)
    try:
        return await asyncio.shield(wrapped)
    except asyncio.CancelledError:
        future.cancel()  # False for a running worker: its finally still owns slot.
        raise


@asynccontextmanager
async def inference_lifespan(app):
    admission = admission_for(app)
    try:
        yield
    finally:
        # Draining a surviving native thread must not block the event loop.
        await asyncio.to_thread(admission.shutdown)
