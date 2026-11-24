"""Krauncher: estimate-only submission through the krauncher client, assay v1
and the GPU ladder, read into the stand's fields."""

import subprocess
import time
import httpx
import krauncher
from datetime import datetime, timezone
from pathlib import Path

from krauncher import KrauncherClient, KrauncherError

from interface import ServiceInterface
from models import Forecast, Task

# The tree the client is imported from (an editable install may point outside
# the stand's own client/): its commit is the client version of a result.
CLIENT = Path(krauncher.__file__).resolve().parent.parent


def client_commit() -> str:
    """The commit of the imported client's tree, "+dirty" when it has
    uncommitted changes: the code that made the forecast, not a copy beside it."""
    head = subprocess.run(["git", "-C", str(CLIENT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(CLIENT), "status", "--porcelain", "--untracked-files=no", "--", "."],
                           capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if head and dirty else "")


def _ensure_data_sources(client: KrauncherClient, sources: list[dict]) -> None:
    """Register the task's data sources on this account when missing."""
    for ds in sources:
        try:
            client.data_source(ds["name"]).info
        except KrauncherError:
            client.data_source(**ds)


def stand_fields(assay: dict, ladder: dict) -> dict:
    """Assay v1 + ladder/2 -> stand fields."""
    wl, model, knobs = assay["workload"], assay["workload"]["model"], assay["workload"]["knobs"]
    w = assay["work"]
    # the compute phase on the reference card (setup, downloads and the warm-up
    # floor are other phases) and its host part (compute phase - gpu_sec): the
    # stand compares the GPU's own time, as published benchmarks measure it
    c = w["reference_sec"] * w["phases_cu"]["compute"] / w["reference_cu"]
    # gpu_sec comes in 0.1 s: a task shorter than that keeps its whole compute phase
    h = max(0.0, c - (w["gpu_sec"] if w.get("gpu_sec") else c))
    return {
        "workload_type": wl["type"],
        "mode": wl["mode"],
        "framework": wl["framework"],
        "inference_engine": wl["runtime"]["inference_engine"],
        "precision": model["precision"],
        "model_name": model["name"],
        "params_billions": model["params_billions"],
        "batch_size": knobs["batch_size"],
        "epochs": knobs["epochs"],
        "dataset_samples": knobs["dataset_samples"],
        # batch passes of the run (training steps, forward / generate calls)
        "steps": (assay["work"] or {}).get("steps"),
        "seq_len": knobs["seq_len"],
        "cpu_only": assay["requirements"]["cpu_only"],
        "min_vram_gb": assay["requirements"]["min_vram_gb"],
        "reference_sec": assay["work"]["reference_sec"],
        "compute_sec": c - h,
        "spread_factor": assay["work"]["spread"]["factor"],
        # the GPU's own part per card: the ladder adds the host part h flat to every
        # card's compute, so the GPU ratio is (ratio x c - h) / (c - h)
        "compute_ratio": {r["gpu_id"]: (r["compute_ratio"] * c - h) / (c - h) for r in ladder["rows"]},
    }


class KrauncherAdapter(ServiceInterface):
    name = "krauncher"

    def capabilities(self) -> set[str]:
        return {"workload_type", "mode", "framework", "inference_engine", "precision", "model_name",
                "params_billions", "batch_size", "epochs", "steps", "dataset_samples", "seq_len",
                "cpu_only", "min_vram_gb", "reference_sec", "compute_sec", "spread_factor", "compute_ratio"}

    async def forecast(self, task: Task) -> Forecast:
        mod = task.variant(self.name)
        client = KrauncherClient(estimate_only=True)
        _ensure_data_sources(client, getattr(mod, "DATA_SOURCES", []))
        handle = await client.task(**mod.OPTIONS)(mod.FUNC)(**mod.KWARGS)
        t0 = time.monotonic()
        job_id = handle.classification.analyzer_job_id
        analyzer = client.analyzer()
        # Stand fields read assay/1 (still served at /jobs/{id}/assay); the
        # ladder comes from the assay/2 path, its ladder/3 rows match ladder/2.
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(f"{analyzer._url}/jobs/{job_id}/assay", headers=analyzer._headers)
            r.raise_for_status()
            assay = r.json()
        ladder = (await client.ladder(await analyzer.assay(job_id)))["ladder"]
        t_assay_ladder = time.monotonic() - t0
        return Forecast(
            service=self.name,
            task=task.name,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            version={
                "calibration_id": ladder["meta"]["calibration_id"],
                "client_commit": client_commit(),
            },
            fields=stand_fields(assay, ladder),
            raw={"assay": assay, "ladder": ladder},
            request={"options": mod.OPTIONS, "kwargs": mod.KWARGS},
            timing={
                # code analysis as the client measured it (submit -> classification)
                "analysis_sec": round(handle.classification.analyzer_time or 0.0, 2),
                "assay_ladder_sec": round(t_assay_ladder, 2),
            },
        )
