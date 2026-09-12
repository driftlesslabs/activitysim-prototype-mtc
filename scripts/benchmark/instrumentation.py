"""Process-local component timing and a strict Sharrow disk-cache guard."""

import json
import multiprocessing
import os
import time
from pathlib import Path


def install(flow_cache=Path("/results/cache/flows")):
    """Install in the model parent and every spawned worker, before model imports."""
    from activitysim.core.workflow import runner

    original = runner.run_named_step
    records = Path(os.environ["BENCH_PHASE_DIR"])

    def timed(name, context, **kwargs):
        started = time.perf_counter()
        succeeded = False
        try:
            result = original(name, context, **kwargs)
            succeeded = True
            return result
        finally:
            finished = time.perf_counter()
            row = {
                "component": name,
                "seconds": finished - started,
                "process": multiprocessing.current_process().name,
                "pid": os.getpid(),
                "succeeded": succeeded,
            }
            # Linux's monotonic clock is shared across processes. Use the
            # supervisor's origin so worker windows align with memory samples.
            if "BENCH_STARTED_MONOTONIC" in os.environ:
                origin = float(os.environ["BENCH_STARTED_MONOTONIC"])
                row.update(
                    start_seconds=started - origin, end_seconds=finished - origin
                )
            # Separate files avoid interleaving writes from concurrent workers.
            with (records / f"components-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    runner.run_named_step = timed
    if os.environ.get("BENCH_STRICT_CACHE") == "1":
        from numba.core.dispatcher import _FunctionCompiler

        compile_original = _FunctionCompiler.compile
        flow_cache = flow_cache.resolve()

        def compile_checked(self, *args, **kwargs):
            # Numba reaches this method only after failing to load a compiled
            # overload from disk. Ordinary ActivitySim/Numba JIT is still allowed.
            filename = Path(self.py_func.__code__.co_filename).resolve()
            if filename.is_relative_to(flow_cache):
                with (records / f"cache-miss-{os.getpid()}.txt").open("a") as stream:
                    stream.write(str(filename) + "\n")
                raise RuntimeError(f"Measured Sharrow flow cache miss: {filename}")
            return compile_original(self, *args, **kwargs)

        _FunctionCompiler.compile = compile_checked
