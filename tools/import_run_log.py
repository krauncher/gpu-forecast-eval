"""Append the actual runs of a Krauncher log to runs/krauncher_runs.csv.

Reference table, not read by report.py: what a task needed and took when it
actually ran on Krauncher, to check the forecasts and the calibration against.
Reads the log of a measurement script that prints, per task, the lines
`--- <task>  forecast N GB ...`, `VRAM_PEAK ...`, `Task <id> done ... (€x)`,
`RESULT ...` and `OUTPUT ...` (cas-client other/lambda_vram_tier24.py); the
`patch` / `override` lines printed before a task are its departures from the
stand's code. A run already in the table (same task_id) is skipped. The log is
copied to runs/logs/ under the name given.

    python tools/import_run_log.py LOG --calibration c-... --script other/lambda_vram_tier24.py --name 2026-10-28_lambda_tier24
"""

import argparse
import ast
import csv
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "runs" / "krauncher_runs.csv"
COLUMNS = ["run_date", "task", "status", "gpu", "gpu_total_gb", "provider", "worker", "task_id",
           "calibration_id", "vram_forecast_gb", "context_gb", "allocated_gb", "reserved_gb",
           "device_used_gb", "download_sec", "pip_sec", "compute_sec", "train_loop_sec", "total_sec",
           "cost_eur", "departures", "script", "log"]
KV = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")


def parse(text: str) -> list[dict]:
    runs, cur, pending = [], None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("patch ", "override ")):
            kind, rest = s.split(" ", 1)
            pending.append(f"{kind}: {rest.split(': ', 1)[1]}")
        elif s.startswith("--- "):
            m = re.match(r"--- (\S+)\s+forecast (\S+) GB", s)
            cur = {"task": "lambdalabs/" + m.group(1), "vram_forecast_gb": m.group(2),
                   "departures": " | ".join(dict.fromkeys(pending))}
            pending = []
            runs.append(cur)
        elif cur is None:
            continue
        elif s.startswith("Executing on "):
            m = re.search(r"\(([^)]+)\)", s)
            cur["provider"] = m.group(1) if m else ""
        elif s.startswith("VRAM_PEAK "):
            kv = dict(KV.findall(s[len("VRAM_PEAK "):]))
            cur.update(gpu=kv["gpu"], gpu_total_gb=kv["total_gb"],
                       **{k: kv[k] for k in ("context_gb", "allocated_gb", "reserved_gb", "device_used_gb")})
        elif s.startswith("Task ") and " done in " in s:
            cur["task_id"] = s.split()[1]
            m = re.search(r"total [\d.]+ KU \(€([\d.]+)\)", s)
            cur["cost_eur"] = m.group(1) if m else ""
        elif s.startswith("RESULT "):
            kv = dict(KV.findall(s.split(" ", 2)[2]))
            cur.update(status=kv["status"], worker=kv["worker"],
                       **{f"{k}_sec": kv[k].rstrip("s") for k in ("download", "pip", "compute", "total")})
        elif s.startswith("OUTPUT "):
            try:
                out = ast.literal_eval(s.split(" ", 2)[2]).get("task_output") or {}
            except (ValueError, SyntaxError):
                out = {}
            cur["train_loop_sec"] = out.get("elapsed_sec", "")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--calibration", required=True, help="calibration_id of the service at the run")
    ap.add_argument("--script", required=True, help="the script that produced the log")
    ap.add_argument("--name", required=True, help="log name in runs/logs/, e.g. 2026-10-28_lambda_tier24")
    args = ap.parse_args()
    log = ROOT / "runs" / "logs" / f"{args.name}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.log, log)
    seen = set()
    if TABLE.exists():
        with TABLE.open() as f:
            seen = {r["task_id"] for r in csv.DictReader(f)}
    new = [r for r in parse(Path(args.log).read_text()) if r.get("task_id") and r["task_id"] not in seen]
    write_head = not TABLE.exists()
    with TABLE.open("a", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        if write_head:
            w.writeheader()
        for r in new:
            w.writerow({**{c: "" for c in COLUMNS}, **r, "run_date": args.name[:10],
                        "calibration_id": args.calibration, "script": args.script,
                        "log": str(log.relative_to(ROOT))})
    print(f"{TABLE.relative_to(ROOT)}: {len(new)} runs added")


if __name__ == "__main__":
    main()
