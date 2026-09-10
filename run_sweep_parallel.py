#!/usr/bin/env python
"""Parallel noise sweep. Parallelizes over shot-chunks, load-balanced across cores."""
import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from sweep_core import run_chunk

CHUNK_FIELDS = ["d", "noise_model", "noise", "chi", "chunk_id",
                "shots", "fails", "seconds", "host", "pid"]


def n_workers():
    """Respect the cgroup, not the machine. os.cpu_count() lies under SLURM."""
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return int(env)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def wilson(fails, shots, z=1.96):
    """Wilson score interval: behaves sensibly at fails=0, unlike normal approx."""
    if shots == 0:
        return (0.0, 1.0)
    phat, z2 = fails / shots, z * z
    denom = 1.0 + z2 / shots
    centre = (phat + z2 / (2 * shots)) / denom
    half = z * math.sqrt(phat * (1 - phat) / shots + z2 / (4 * shots * shots)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class Point:
    """Accounting for one (d, chi, noise) parameter point."""

    def __init__(self, noise, args):
        self.noise = noise
        self.args = args
        self.shots = self.fails = self.cpu_seconds = 0
        self.launched = 0        # chunk_id counter; also fixes chunk sizes
        self.inflight_shots = 0
        self.done = False

    def _size(self, index):
        """Geometric ramp: small chunks find high error rates fast without
        overshooting; large chunks amortize scheduling for low-p points."""
        return min(self.args.max_chunk, self.args.min_chunk * (2 ** index))

    def wants_more(self):
        if self.done or self.fails >= self.args.nfail:
            return False
        return self.shots + self.inflight_shots < self.args.max_shots

    def next_task(self):
        size = self._size(self.launched)
        budget = self.args.max_shots - (self.shots + self.inflight_shots)
        size = min(size, budget)
        if size <= 0:
            return None
        task = (self.args.d, self.args.noise_model, self.noise, self.args.chi,
                size, self.launched, self.args.seed)
        self.launched += 1
        self.inflight_shots += size
        return task

    def absorb(self, rec, inflight=True):
        self.shots += rec["shots"]
        self.fails += rec["fails"]
        self.cpu_seconds += rec["seconds"]
        if inflight:
            self.inflight_shots -= rec["shots"]
        self.launched = max(self.launched, rec["chunk_id"] + 1)


def load_completed(path, args):
    """Resume support: chunk sizes are a deterministic function of chunk_id,
    so replaying the log reconstructs state exactly."""
    seen = {}
    if not path.exists():
        return seen
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if (int(row["d"]) != args.d or int(row["chi"]) != args.chi
                    or row["noise_model"] != args.noise_model):
                continue
            rec = {k: row[k] for k in CHUNK_FIELDS}
            rec.update(shots=int(row["shots"]), fails=int(row["fails"]),
                       chunk_id=int(row["chunk_id"]), seconds=float(row["seconds"]),
                       noise=float(row["noise"]))
            seen.setdefault(rec["noise"], []).append(rec)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--chi", type=int, default=6)
    ap.add_argument("--noise_model", default="depolarize")
    ap.add_argument("--noises", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--nfail", type=int, default=1000,
                    help="stop a point once it reaches this many failures")
    ap.add_argument("--max_shots", type=int, default=2_000_000,
                    help="hard per-point shot cap; low-p points will hit this")
    ap.add_argument("--min_chunk", type=int, default=500)
    ap.add_argument("--max_chunk", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20240001)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"d{args.d}_chi{args.chi}_{args.noise_model}"
    chunk_log = args.outdir / f"chunks_{tag}.csv"
    summary = args.outdir / f"summary_{tag}.csv"

    workers = args.workers or n_workers()
    points = {p: Point(p, args) for p in args.noises}

    resumed = load_completed(chunk_log, args)
    for p, recs in resumed.items():
        if p in points:
            for rec in recs:
                points[p].absorb(rec, inflight=False)
    if resumed:
        print(f"[resume] replayed {sum(len(v) for v in resumed.values())} chunks",
              flush=True)

    fresh = not chunk_log.exists()
    log_fh = open(chunk_log, "a", newline="")
    writer = csv.DictWriter(log_fh, fieldnames=CHUNK_FIELDS)
    if fresh:
        writer.writeheader()

    t_start = time.perf_counter()
    completed = 0

    # spawn: workers re-import sweep_core, so the BLAS pinning actually applies.
    # fork would inherit a possibly-already-initialized threaded BLAS.
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        pending = {}

        def top_up():
            """Keep the pool saturated, cycling over points that still need shots."""
            while len(pending) < 2 * workers:  # 2x depth hides task turnaround
                candidates = [pt for pt in points.values() if pt.wants_more()]
                if not candidates:
                    return
                pt = min(candidates, key=lambda q: q.shots + q.inflight_shots)
                task = pt.next_task()
                if task is None:
                    pt.done = True
                    continue
                pending[pool.submit(run_chunk, task)] = pt

        top_up()
        while pending:
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                pt = pending.pop(fut)
                rec = fut.result()          # re-raises worker exceptions here
                pt.absorb(rec)
                writer.writerow(rec)
                log_fh.flush()
                completed += 1

            done_pts = sum(1 for q in points.values()
                           if not q.wants_more() and q.inflight_shots == 0)
            print(f"[{time.perf_counter() - t_start:8.1f}s] chunks={completed} "
                  f"points_done={done_pts}/{len(points)} "
                  + " ".join(f"p={q.noise:g}:{q.fails}/{q.shots}"
                             for q in points.values()), flush=True)
            top_up()

    log_fh.close()

    with open(summary, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["d", "noise_model", "noise", "chi", "shots", "fails",
                    "error_rate", "ci_low", "ci_high", "reached_nfail", "cpu_seconds"])
        for p in args.noises:
            pt = points[p]
            rate = pt.fails / pt.shots if pt.shots else float("nan")
            lo, hi = wilson(pt.fails, pt.shots)
            w.writerow([args.d, args.noise_model, p, args.chi, pt.shots, pt.fails,
                        f"{rate:.6g}", f"{lo:.6g}", f"{hi:.6g}",
                        int(pt.fails >= args.nfail), f"{pt.cpu_seconds:.1f}"])

    wall = time.perf_counter() - t_start
    cpu = sum(q.cpu_seconds for q in points.values())
    print(f"\nwall={wall:.1f}s  worker_cpu={cpu:.1f}s  "
          f"speedup={cpu / wall:.1f}x on {workers} workers", flush=True)
    for p in args.noises:
        pt = points[p]
        flag = "" if pt.fails >= args.nfail else "  <-- HIT max_shots, treat as bound"
        print(f"  p={p:<8g} {pt.fails:>7d}/{pt.shots:<10d} "
              f"= {pt.fails / max(pt.shots,1):.3e}{flag}")


if __name__ == "__main__":
    main()