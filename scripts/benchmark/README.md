# Linux Docker benchmark

Run `scripts/production-benchmark.py` from any directory using Python 3.10+ and a
running Linux Docker engine with **cgroup v2** and `memory.peak` support. Docker
Desktop works on macOS. Allocate enough RAM to its Linux VM for the memory limit,
plus VM overhead. The host does not need ActivitySim, Sharrow, pandas, or plotting
packages. Reports contain inline SVG and work offline.

```bash
python scripts/production-benchmark.py \
  --activitysim-commit <full-40-character-activitysim-SHA> \
  --sharrow-commit <full-40-character-sharrow-SHA> \
  --multiprocess --processes 4 --sharrow --households 100000 \
  --data-dir /absolute/path/to/data_full \
  --memory 24g --shm-size 12g \
  --output-dir /absolute/path/to/experiments/mp4-sharrow \
  --label '4 workers, Sharrow'
```

Replace the SHA placeholders with actual full Git object IDs. The Dockerfile
fetches each exact object from the official ActivitySim organization repositories,
checks the resolved ID, and installs both packages together. Revisions must
support Python 3.11 and the ActivitySim `workflow.State` API (ActivitySim 1.4+);
incompatible historical commits fail with their build/model logs preserved.
The benchmark uses Numba's private `_FunctionCompiler.compile` hook to reject
Sharrow cache misses; dependency/API changes may require adapting that hook.

Use `--single-process` (default) for serial execution, `--no-sharrow` to disable
Sharrow, and `--households 0` for the full population. Sharrow is still installed
at the specified revision when disabled. `--processes` is the worker count in
sliced MP phases, not a cap on the coordinator plus all child processes. Serial
initialization/finalization phases have one observation. Defaults are 1,000
households, Sharrow enabled, 16 GiB memory, 8 GiB `/dev/shm`, and 0.5 second sampling.
Choose limits appropriate for the data and worker count; **chunking is disabled**.
That is the base model default. Use `--config-overlay configs_explicit_chunk` to
activate the supplied explicit row limits. Additional overlay directories are
searched in the listed order before `configs_mp` and `configs`, and are copied
into each experiment's model snapshot. The runner honors their chunking mode.
`--platform linux/amd64` or `linux/arm64` can select an architecture; omitted means
Docker's native architecture. Emulated and native timings are not equivalent.

## Input data

Use the existing full CSV/OMX data directory (`data_full` by default). The smaller
repository `data` directory contains Parquet tables and is not the benchmark's
input preset. Obtain the full dataset from the model's
[v1.3.4 data release](https://github.com/ActivitySim/activitysim-prototype-mtc/releases/download/v1.3.4/data_full.tar.zst)
and extract it into a directory containing `households.csv`, `persons.csv`,
`land_use.csv`, and `skims.omx`. The archive's SHA-256 is
`b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3`.
Input data are mounted read-only. Do not modify them while an experiment runs.

Each experiment snapshots `configs` and `configs_mp` from this checkout, records
the model Git revision/status and configuration hashes, input file sizes/mtimes,
Docker hardware information, image ID and installed package versions. Sizes and
mtimes are provenance hints, not content hashes of the large input dataset.
`effective-settings.json` preserves the complete resolved ActivitySim settings.
The image records installed dependencies in `pip-freeze.txt`; resolving a fresh
image later can select different transitive dependencies. Retain the image ID
and compare dependency manifests when attributing changes to source revisions.

## Cache and measurement protocol

With Sharrow enabled, the script first runs the **entire model with the same
sample, seed and worker layout** in a separate container. This costs a full model
run but avoids guessing which flows a tiny training sample exercises. The fresh
experiment-owned cache is shared at the same absolute path with the measured
container. Measurement refuses to compile any generated flow overload on a disk
cache miss and marks the experiment failed. Ordinary non-flow Numba compilation
and disk-cache loading remain part of measured runtime. No warmup runs when
Sharrow is disabled. Both phases use `sharrow=require` when enabled; that setting
alone does not prohibit compilation, which is why the additional guard exists.

Optionally use `--cache-from /path/to/previous-experiment` to seed the flow cache
before warmup. Both package commits and the installed dependency manifest must
match. A complete warmup still runs with the new settings to build any missing
flow signatures; the measured run still rejects compilation. Only flow artifacts
are reused, not model outputs or checkpoints.

Both modes use the same base component list, excluding the diagnostic
`track_skim_usage`. Shadow pricing, trace households/ODs, and profiling are
disabled; the RNG seed is zero. Pipeline checkpoint writing stays enabled.
Numba, BLAS, OpenMP and NumExpr threads are limited to one; Dask uses its
synchronous scheduler. MP slicing comes from `configs_mp`.

The supervisor samples **whole-container `memory.current`** and kernel
`memory.peak` throughout the model process lifetime. Shared pages are charged
once, unlike summing worker RSS. The total includes anonymous pages, shared
memory, file cache, kernel memory and the small supervisor. `memory.csv` also
records shared/file/anonymous bytes and swap; these columns overlap and must
not be added together. Memory+swap equals the memory limit, disabling container
swap. `/dev/shm` is a capacity within that limit, not additional allocated RAM.
Kernel peak includes container startup but excludes the separate cache-build
container and post-run summary work. Very short peaks can be absent from the
sampled line but still appear in the kernel peak value. On a fatal OOM the last
sample is a lower bound; Docker's OOM state marks the run unsuccessful.

Containers share the Linux VM's filesystem page cache. This is not a cold-input
I/O benchmark, and already-cached file pages can be charged outside the current
container. Keep VM resources, data and architecture consistent across experiments.

Component timings wrap each actual workflow step in every process, excluding
checkpoint writes between steps. The report displays mean, **population SD**,
observation count and maximum. SD describes observed worker imbalance, not
statistical confidence across repeated experiments. The mean is not the wall
clock duration of an MP phase. Overall elapsed time additionally includes Python
startup, imports, coordination and checkpoint I/O. Failed components are retained
in raw JSONL but omitted from the successful-execution aggregates.

Input counts describe all source households, persons and zones. Final household
and person counts describe the realized sample. Output summaries count tours,
trips and joint participants and show available purpose/mode/category counts;
land-use summaries include population, households and employment totals.

## Outputs and comparisons

`--output-dir` must name a new directory to avoid mixing experiments or caches.
It contains `report.html`, normalized `report.json`, `experiment.json`, build and
console logs, source/config snapshots, `pip-freeze.txt`, `cache/`, and separate
`warmup/` and `measured/` folders. Each phase holds raw component JSONL, memory CSV,
resolved settings, status, input/output summaries, and ActivitySim's `output/`.
Build and warmup are excluded from measured results. Failed builds and runs still
produce a report and retain logs. Containers are removed; images and experiment
artifacts remain for inspection. Model output, checkpoints and warmup output can
be large.

Add previous experiment directories to a new run:

```bash
python scripts/production-benchmark.py \
  --activitysim-commit <SHA> --sharrow-commit <SHA> \
  --single-process --no-sharrow --households 100000 \
  --output-dir /absolute/path/to/experiments/serial \
  --compare /absolute/path/to/experiments/mp4-sharrow
```

Regenerate a comparison without running Docker or a model:

```bash
python scripts/production-benchmark.py --report-only \
  --compare /absolute/path/to/experiments/mp4-sharrow /absolute/path/to/experiments/serial \
  --output-dir /absolute/path/to/comparison.html
```

Memory panels are side-by-side on common axes; runtime bars are grouped by
component and experiment. Green table cells identify the fastest successful
experiment's mean (ties included). Inspect settings and realized sample sizes
before interpreting comparisons with different workloads.

The **Highlight component** dropdown updates every memory panel together. Each
worker's execution window is a separate translucent band: overlapping workers
produce darker shading, while gaps with no active worker remain unshaded. Hover
over a band for its worker, elapsed range, and timing source. Select **None** to
clear highlighting. The selector works offline and requires JavaScript.

New runs record `start_seconds` and `end_seconds` in the component JSONL using
the supervisor's shared monotonic clock, aligning directly with memory samples.
Reports regenerated from older runs reconstruct approximate windows from
`console.log` completion messages where possible. Those older windows are
explicitly labeled approximate: they can include checkpoint time and logging
delay and have a small clock-origin offset. If no suitable timestamps or log
messages exist, the chart says that window data is unavailable for that component.
The original runtime aggregates are unchanged by this fallback.

## Development checks

In an environment containing ActivitySim, Numba, pandas and pyarrow:

```bash
python -m unittest discover -s scripts/benchmark -p 'test_*.py' -v
pre-commit run --config scripts/benchmark/pre-commit.yaml \
  --files scripts/production-benchmark.py scripts/benchmark/instrumentation.py \
  scripts/benchmark/worker.py scripts/benchmark/test_benchmark.py
```

The local pre-commit hooks require `ruff` on PATH. Tests cover shared-memory
accounting, worker aggregation, failed-run exclusion, HTML escaping, table
summaries and real Numba disk-cache hit/miss behavior across fresh interpreters.
They also check shared-clock worker timestamps, legacy log recovery, overlapping
and disjoint windows, and selector behavior against generated SVG metadata.
Install Node.js to run the offline JavaScript controller test (otherwise it skips).

The measurement approach was informed by the
[Lighthouse production benchmark](https://github.com/wsp-sag/lighthouse/blob/go-sharrow/scripts/production-benchmark.py)
and its [Dockerfile](https://github.com/wsp-sag/lighthouse/blob/go-sharrow/scripts/production-benchmark.Dockerfile).
