"""Measurement/report regression tests; no running Docker daemon required."""

import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "benchmark", HERE.parent / "production-benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
spec = importlib.util.spec_from_file_location("worker", HERE / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class BenchmarkTests(unittest.TestCase):
    def make_run(self, root, label, times, success=True):
        """Construct raw artifacts with the same schema emitted by workers."""
        root.mkdir()
        phase = root / "measured"
        phase.mkdir()
        benchmark.write_json(
            root / "experiment.json", {"schema_version": 1, "label": label}
        )
        benchmark.write_json(
            phase / "status.json", {"returncode": 0, "elapsed_seconds": 9}
        )
        benchmark.write_json(
            phase / "docker-state.json", {"ExitCode": 0, "OOMKilled": not success}
        )
        for i, duration in enumerate(times):
            (phase / f"components-{i}.jsonl").write_text(
                json.dumps(
                    {
                        "component": "auto_ownership",
                        "seconds": duration,
                        "succeeded": True,
                    }
                )
                + "\n"
            )
        # A duplicate locutor CSV must never contribute another observation.
        (phase / "timing_log.csv").write_text(
            "model_name,seconds\nauto_ownership,999\n"
        )
        with (phase / "memory.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["elapsed_seconds", "current_bytes", "peak_bytes"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"elapsed_seconds": 0, "current_bytes": 100, "peak_bytes": 100},
                    {"elapsed_seconds": 9, "current_bytes": 200, "peak_bytes": 300},
                ]
            )
        return root

    def test_worker_statistics_and_failed_run_winners(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = self.make_run(root / "a", "A <script>", [2, 4])
            b = self.make_run(root / "b", "B", [1], success=False)
            run = benchmark.load_run(a)
            self.assertEqual(
                run["components"]["auto_ownership"],
                {"n": 2, "mean": 3, "sd": 1, "maximum": 4},
            )
            self.assertEqual(run["peak"], 300)
            benchmark.report([a, b], root / "report.html")
            html = (root / "report.html").read_text()
            self.assertIn('class="fastest">3.000', html)
            self.assertNotIn('class="fastest">1.000', html)
            self.assertIn("A &lt;script&gt;", html)
            self.assertIn("FAILED / INCOMPLETE", html)

    def test_incomplete_run_report_and_cache_miss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = self.make_run(root / "a", "A", [1])
            (a / "measured/cache-miss-1.txt").write_text("missing overload")
            self.assertFalse(benchmark.load_run(a)["valid"])
            b = root / "build-failed"
            b.mkdir()
            benchmark.write_json(
                b / "experiment.json", {"schema_version": 1, "label": "failed build"}
            )
            benchmark.report([a, b], root / "report.html")
            self.assertEqual(benchmark.load_run(b)["components"], {})

    def test_shared_memory_is_not_added_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in {
                "memory.current": "1000",
                "memory.peak": "1500",
                "memory.swap.current": "0",
                "memory.stat": "anon 300\nfile 600\nshmem 400\n",
            }.items():
                (root / name).write_text(value)
            row = worker.sample_memory(root, 2)
            self.assertEqual(row["current_bytes"], 1000)
            self.assertEqual(row["shared_bytes"], 400)

    def test_exact_and_legacy_worker_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            phase = Path(temporary)
            (phase / "console.log").write_text(
                "[00:05.00] INFO: mp_households_0 choice : 3.0 seconds (0.1 minutes)\n"
                "[00:09.00] INFO: mp_households_0 choice : 2.0 seconds (0.0 minutes)\n"
                "[01:02:03.00] Level 25: time to execute run.serial : 1:03\n"
            )
            base = {
                "component": "choice",
                "process": "mp_households_0",
                "succeeded": True,
            }
            rows = [
                dict(base, start_seconds=1.0, end_seconds=4.0),
                dict(base, seconds=2.0),
                dict(
                    base,
                    process="mp_households_1",
                    start_seconds=2.0,
                    end_seconds=5.0,
                    succeeded=False,
                ),
                {"component": "serial", "process": "MainProcess", "succeeded": True},
                dict(base, component="unavailable"),
            ]
            windows = benchmark.component_windows(rows, phase)
            self.assertEqual(
                [(w["start_seconds"], w["end_seconds"]) for w in windows],
                [(1, 4), (7, 9), (2, 5), (3660, 3723)],
            )
            self.assertEqual(windows[0]["source"], "recorded monotonic clock")
            self.assertEqual(windows[1]["source"], "approximate completion log")
            self.assertFalse(windows[2]["succeeded"])

    @unittest.skipUnless(
        shutil.which("node"), "Node.js needed for offline controller test"
    )
    def test_selector_preserves_overlap_and_gaps_across_charts(self):
        """Execute the emitted controller against the emitted SVG band metadata."""

        class Bands(HTMLParser):
            def __init__(self):
                super().__init__()
                self.panels = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "div" and attrs.get("class") == "memory-panel":
                    self.panels.append([])
                if tag == "rect" and attrs.get("class") == "component-window":
                    self.panels[-1].append(attrs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = self.make_run(root / "a", "A", [1])
            b = self.make_run(root / "b", "B", [1])
            name = 'choice <&"'
            for i, (start, end) in enumerate(((1, 4), (2, 5), (7, 8))):
                row = {
                    "component": name,
                    "process": f"worker_{i}",
                    "succeeded": True,
                    "seconds": end - start,
                    "start_seconds": start,
                    "end_seconds": end,
                }
                (a / "measured" / f"components-{i}.jsonl").write_text(
                    json.dumps(row) + "\n"
                )
            benchmark.report([a, b], root / "report.html")
            document = (root / "report.html").read_text()
            parsed = Bands()
            parsed.feed(document)
            bands = parsed.panels[0]
            self.assertEqual(len(bands), 3)
            # Overlapping windows remain separate; the gap from 5s to 7s is clear.
            self.assertLess(
                float(bands[1]["x"]), float(bands[0]["x"]) + float(bands[0]["width"])
            )
            self.assertGreater(
                float(bands[2]["x"]), float(bands[1]["x"]) + float(bands[1]["width"])
            )
            script = re.search(r"<script>(.*?)</script>", document, re.DOTALL)[1]
            controller_test = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
let change;
const selector = {value: '', addEventListener(type, fn) {assert.equal(type, 'change'); change = fn;}};
const panels = input.panels.map(attrs => {
  const bands = attrs.map(a => ({dataset: {component: a['data-component'], source: a['data-source']}, style: {display: 'none'}}));
  const status = {textContent: ''};
  return {bands, status, querySelectorAll() {return bands;}, querySelector() {return status;}};
});
vm.runInNewContext(input.script, {document: {
  getElementById(id) {assert.equal(id, 'memory-component'); return selector;},
  querySelectorAll() {return panels;}
}});
selector.value = input.name;
change();
assert.equal(panels[0].bands.filter(b => b.style.display === '').length, 3);
assert.match(panels[0].status.textContent, /3 worker execution windows/);
assert.match(panels[1].status.textContent, /No execution-window data/);
selector.value = '';
change();
assert.ok(panels.every(p => p.bands.every(b => b.style.display === 'none')));
"""
            subprocess.run(
                [shutil.which("node"), "-e", controller_test],
                input=json.dumps(
                    {"panels": parsed.panels, "name": name, "script": script}
                ),
                text=True,
                check=True,
            )

    def test_commit_and_interval_validation(self):
        import argparse

        for value in ("main", "1234567", "x" * 40):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark.commit(value)
        for value in ("nan", "inf", "0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark.positive(value)

    def test_table_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "households.csv").write_text("HHID,PERSONS\n1,2\n2,1\n")
            (root / "land_use.csv").write_text("TAZ,TOTPOP,TOTEMP\n1,20,10\n2,30,15\n")
            (root / "trips.csv").write_text(
                "trip_id,trip_mode\n1,WALK\n2,WALK\n3,SOV\n"
            )
            summary = worker.table_summary(root)
            self.assertEqual(summary["households"]["rows"], 2)
            self.assertEqual(summary["land_use"]["totals"]["TOTPOP"], 50)
            self.assertEqual(
                summary["trips"]["categories"]["trip_mode"], {"WALK": 2, "SOV": 1}
            )

    def test_real_numba_disk_hit_allowed_and_miss_blocked(self):
        """A fresh interpreter must load warmed overloads but reject new ones."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "generated_flow.py").write_text(
                "from numba import njit\n@njit(cache=True)\ndef f(x):\n    return x + 1\n"
            )
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join([str(HERE), str(root)]),
                BENCH_PHASE_DIR=str(root),
                BENCH_STRICT_CACHE="1",
            )
            warm = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from generated_flow import f; assert f(1) == 2",
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(warm.returncode, 0, warm.stderr)
            script = f"from pathlib import Path; from instrumentation import install; install(Path({str(root)!r})); from generated_flow import f; "
            hit = subprocess.run(
                [sys.executable, "-c", script + "assert f(1) == 2"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(hit.returncode, 0, hit.stderr)
            miss = subprocess.run(
                [sys.executable, "-c", script + "f(1.5)"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(miss.returncode, 0)
            self.assertIn("Measured Sharrow flow cache miss", miss.stderr)
            self.assertTrue(list(root.glob("cache-miss-*.txt")))

    def test_spawned_workers_each_record_components(self):
        """Exercise the same top-level hook installation used by MP workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "spawn_check.py"
            script.write_text(
                "from activitysim.core.workflow import runner\n"
                "runner.run_named_step = lambda name, context: 'ok'\n"
                "import worker\n"
                "import multiprocessing\n"
                "def task():\n"
                "    assert runner.run_named_step('smoke', {}) == 'ok'\n"
                "if __name__ == '__main__':\n"
                "    ctx = multiprocessing.get_context('spawn')\n"
                "    processes = [ctx.Process(target=task) for _ in range(2)]\n"
                "    for p in processes: p.start()\n"
                "    for p in processes:\n"
                "        p.join()\n"
                "        assert p.exitcode == 0\n"
            )
            env = dict(
                os.environ,
                PYTHONPATH=str(HERE),
                BENCH_MODEL="1",
                BENCH_PHASE_DIR=str(root),
                BENCH_STRICT_CACHE="0",
                BENCH_STARTED_MONOTONIC=str(time.perf_counter()),
            )
            result = subprocess.run(
                [sys.executable, str(script)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [
                json.loads(path.read_text()) for path in root.glob("components-*.jsonl")
            ]
            self.assertEqual(len({row["pid"] for row in records}), 2)
            self.assertTrue(
                all(row["succeeded"] and row["component"] == "smoke" for row in records)
            )
            for row in records:
                self.assertGreater(row["start_seconds"], 0)
                self.assertAlmostEqual(
                    row["end_seconds"] - row["start_seconds"], row["seconds"]
                )


if __name__ == "__main__":
    unittest.main()
