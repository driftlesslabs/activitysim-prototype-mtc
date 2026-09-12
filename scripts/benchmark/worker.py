"""Linux container supervisor; the model runs in a separate process tree."""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# multiprocessing's spawn imports this file again as __mp_main__.
if os.environ.get("BENCH_MODEL") == "1":
    from instrumentation import install

    install()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_model(spec, phase):
    """Use identical samples, process layout and cache paths in both phases."""
    import activitysim.abm  # noqa: F401 -- register model components
    import yaml
    from activitysim.core.workflow import State

    configs = (
        ["/model/configs_mp", "/model/configs"]
        if spec["multiprocess"]
        else ["/model/configs"]
    )
    base = yaml.safe_load(Path("/model/configs/settings.yaml").read_text())
    # Both modes run the same component list (the repository's MP overlay differs).
    settings = {
        "households_sample_size": spec["households"],
        "multiprocess": spec["multiprocess"],
        "num_processes": spec["processes"],
        "sharrow": "require" if spec["sharrow"] else False,
        "chunk_size": 0,
        "chunk_training_mode": "disabled",
        "use_shadow_pricing": False,
        "trace_hh_id": None,
        "trace_od": None,
        "resume_after": None,
        "instrument": False,
        "memory_profile": False,
        "expression_profile": False,
        "checkpoints": True,
        "models": [name for name in base["models"] if name != "track_skim_usage"],
        "rng_base_seed": 0,
        "benchmarking": False,
    }
    state = State.make_default(
        working_dir=Path("/model"),
        configs_dir=configs,
        data_dir=Path("/data"),
        output_dir=phase / "output",
        cache_dir=Path("/results/cache/model"),
        settings=settings,
    )
    state.filesystem.sharrow_cache_dir = Path("/results/cache/flows")
    # MP rebuilds FileSystem from settings rather than forwarding this field.
    state.settings.sharrow_cache_dir = str(state.filesystem.sharrow_cache_dir)
    state.set("imported_extensions", ())
    state.set("run_timestamp", "benchmark")
    state.set("run_id", str(state.tracing.run_id))
    state.logging.config_logger()
    write_json(
        phase / "effective-settings.json", state.settings.model_dump(mode="json")
    )
    state.run.all(resume_after=None)
    if not spec["multiprocess"]:
        state.checkpoint.close_store()


def table_summary(directory, prefix=""):
    """Stream input/output tables so full-population summaries need bounded RAM."""
    import pandas as pd
    import pyarrow.parquet as pq

    result = {}
    for name in (
        "households",
        "persons",
        "land_use",
        "tours",
        "trips",
        "joint_tour_participants",
    ):
        path = directory / f"{prefix}{name}.csv"
        if path.exists():
            chunks = pd.read_csv(path, chunksize=100_000)
        else:
            path = directory / f"{prefix}{name}.parquet"
            if not path.exists():
                continue
            chunks = (
                batch.to_pandas() for batch in pq.ParquetFile(path).iter_batches()
            )
        info = {"rows": 0, "totals": {}, "categories": {}}
        for chunk in chunks:
            info["rows"] += len(chunk)
            for col in ("TOTPOP", "TOTHH", "TOTEMP"):
                if col in chunk:
                    info["totals"][col] = info["totals"].get(col, 0) + float(
                        chunk[col].sum()
                    )
            for col in (
                "tour_category",
                "tour_type",
                "tour_mode",
                "trip_mode",
                "primary_purpose",
            ):
                if col in chunk:
                    counts = info["categories"].setdefault(col, {})
                    for value, count in chunk[col].value_counts(dropna=False).items():
                        counts[str(value)] = counts.get(str(value), 0) + int(count)
        result[name] = info
    return result


def sample_memory(root, elapsed):
    """cgroup v2 charges shared pages once; file includes shmem, not vice versa."""
    stat = dict(
        line.split() for line in (root / "memory.stat").read_text().splitlines()
    )
    return {
        "elapsed_seconds": elapsed,
        "current_bytes": int((root / "memory.current").read_text()),
        "peak_bytes": int((root / "memory.peak").read_text()),
        "swap_bytes": int((root / "memory.swap.current").read_text()),
        "anonymous_bytes": int(stat.get("anon", 0)),
        "file_bytes": int(stat.get("file", 0)),
        "shared_bytes": int(stat.get("shmem", 0)),
    }


def supervise(spec, phase):
    """Measure only the model subprocess lifetime; summarize after sampling ends."""
    root = Path("/sys/fs/cgroup")
    if not (root / "memory.peak").exists():
        raise RuntimeError("A private cgroup v2 with memory.peak is required")
    (phase / "output").mkdir()
    for name in ("flows", "model"):
        Path(f"/results/cache/{name}").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BENCH_MODEL="1", BENCH_PHASE_DIR=str(phase))
    env["BENCH_STRICT_CACHE"] = (
        "1" if spec["sharrow"] and phase.name == "measured" else "0"
    )
    started = time.perf_counter()
    env["BENCH_STARTED_MONOTONIC"] = str(started)
    with (phase / "memory.csv").open("w", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_memory(root, 0).keys())
        writer.writeheader()
        writer.writerow(sample_memory(root, 0))
        process = subprocess.Popen(
            [sys.executable, __file__, "model", str(phase)], env=env
        )
        while True:
            writer.writerow(sample_memory(root, time.perf_counter() - started))
            if process.poll() is not None:
                break
            try:
                process.wait(timeout=spec["interval"])
            except subprocess.TimeoutExpired:
                pass
    write_json(
        phase / "status.json",
        {
            "returncode": process.returncode,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    # This separate process keeps pandas/Arrow and summary allocations out of the
    # measured lifetime; the host report uses only the recorded cgroup samples.
    subprocess.run([sys.executable, __file__, "summary", str(phase)], check=True)
    return process.returncode


if __name__ == "__main__":
    mode, phase_arg = sys.argv[1:]
    phase = Path(phase_arg)
    spec = json.loads(Path("/results/experiment.json").read_text())
    if mode == "model":
        run_model(spec, phase)
    elif mode == "summary":
        write_json(phase / "input-summary.json", table_summary(Path("/data")))
        write_json(
            phase / "output-summary.json", table_summary(phase / "output", "final_")
        )
    else:
        sys.exit(supervise(spec, phase))
