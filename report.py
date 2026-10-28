"""Report over the frozen forecasts in results/, per service.

For every (service, task) the latest result is taken (optionally only those of
one service version, e.g. a Krauncher calibration_id) and compared with the
task's description (tasks/<source>/<task>.json). Levels:

0. forecast time — wall clock from request to the normalised forecast,
   measured by the stand for every service (plus the service's own parts);
1. classification / assay fields — match per field;
2. reference-card time — forecast / reference per task: centre (geomean),
   typical deviation (median |log|), share inside the issued spread;
3. VRAM — forecast / measured peak; forecasts below the peak (an OOM risk);
   forecasts above a measured upper bound (`vram_gb_at_most`);
4. ladder — compute_ratio per GPU: error (median and p90 of |log|), card order
   (Spearman), time regret of the GPU the forecast calls fastest; measured
   GPUs the service gave no forecast for.

Levels 2-4 are reported per measurement set of the descriptions. Every value
gets a status — good (accurate), warning, critical — by the thresholds in
scoring.py. Output: report.json (machine-readable), report.md, report.html
(for viewing) in --out.

    python report.py [--service krauncher] [--version <id>] [--measurement <id>] [--out DIR]
"""

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import scoring
from scoring import (MEASURED, REFERENCE_GPU, TIME_GOOD, anchor_time, field_status, ladder_pairs, ratio_status,
                     time_status, vram_status)

ROOT = Path(__file__).parent


# --------------------------------------------------------------------- data

def latest(service: str | None, version: str | None) -> dict[tuple[str, str], dict]:
    """(service, task) -> its latest result (results/<service>/...)."""
    out: dict[tuple[str, str], dict] = {}
    for f in sorted((ROOT / "results").glob("*/**/*.json")):
        r = json.loads(f.read_text())
        if "service" not in r or (service and r["service"] != service):
            continue
        if version and version not in r["version"].values():
            continue
        key = (r["service"], r["task"])
        if key not in out or r["created_utc"] > out[key]["created_utc"]:
            out[key] = r
    return out


def _gm(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


def _q(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def _r(x, nd=3):
    if x is None:
        return None
    # a non-zero value that would round to 0 keeps 3 significant digits
    return round(x, nd) if round(x, nd) or not x else float(f"{x:.3g}")


def _typ(ratios):
    return _r(math.exp(statistics.median(abs(math.log(x)) for x in ratios)))


def analyze(results: dict, descs: dict, measurement: str | None = None) -> dict:
    """Everything the report shows, as data."""
    mids = [measurement] if measurement else list(dict.fromkeys(
        m["id"] for d in descs.values() for m in d.get("measurements", [])))
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {
            "time_good": scoring.TIME_GOOD, "time_warning": scoring.TIME_WARN,
            "ratio_good": scoring.RATIO_GOOD, "ratio_warning": scoring.RATIO_WARN,
            "vram_over_warning": scoring.VRAM_OVER_WARN, "vram_bound_far": scoring.VRAM_BOUND_FAR, "params_tolerance": scoring.PARAMS_TOL,
            "field_near": scoring.FIELD_NEAR,
            "fields_good": scoring.FIELDS_GOOD, "fields_warning": scoring.FIELDS_WARN,
        },
        "services": [],
    }
    for service in sorted({s for s, _ in results}):
        rs = {t: r for (s, t), r in results.items() if s == service and t in descs}
        svc = {"service": service, "tasks": sorted(rs),
               "versions": sorted({json.dumps(r["version"], sort_keys=True) for r in rs.values()}),
               "result_created_utc": {t: rs[t]["created_utc"] for t in sorted(rs)}}

        # 0. forecast time
        timed = {t: rs[t]["timing"] for t in sorted(rs) if (rs[t].get("timing") or {}).get("forecast_sec")}
        ts = [v["forecast_sec"] for v in timed.values()]
        svc["forecast_time"] = {
            "median_sec": _r(statistics.median(ts), 2) if ts else None,
            "p90_sec": _r(_q(ts, 0.9), 2) if ts else None,
            "max_sec": _r(max(ts), 2) if ts else None,
            "tasks": timed,
        }

        # 1. classification / assay fields
        fields: dict[str, dict] = {}
        for t in sorted(rs):
            exp, got = descs[t]["fields"], rs[t]["fields"]
            for f, w in exp.items():
                if f in MEASURED or f not in got:
                    continue
                fields.setdefault(f, {"tasks": {}})["tasks"][t] = {
                    "expected": w, "forecast": got[f], "status": field_status(f, w, got[f])}
        for v in fields.values():
            v["match"] = sum(1 for x in v["tasks"].values() if x["status"] == "good")
            v["total"] = len(v["tasks"])
        order = ("workload_type", "mode", "framework", "inference_engine", "precision", "params_billions", "batch_size",
                 "epochs", "steps", "dataset_samples", "seq_len", "cpu_only")
        svc["fields"] = {f: fields[f] for f in sorted(fields, key=lambda f: (order.index(f) if f in order else len(order), f))}

        # 2-4. per measurement set
        svc["measurements"] = []
        for mid in mids:
            def meas(t, mid=mid):
                return next((m for m in descs[t].get("measurements", []) if m["id"] == mid), {})
            have = [t for t in sorted(rs) if meas(t)]
            if not have:
                continue
            ms = {"id": mid, "independent": sorted({bool(meas(t).get("independent")) for t in have}),
                  "tasks": have}

            # 2. reference-card time
            rows = []
            for t in have:
                m, g = meas(t).get("reference_sec"), rs[t]["fields"].get("reference_sec")
                if m and g:
                    sp = rs[t]["fields"].get("spread_factor") or 1.0
                    rows.append({"task": t, "measured": m, "forecast": g, "ratio": _r(g / m),
                                 "spread_factor": sp, "inside_spread": 1 / sp <= m / g <= sp,
                                 "status": time_status(g, m)})
            if rows:
                ms["reference_time"] = {"centre": _r(_gm([x["ratio"] for x in rows])),
                                        "typical_deviation": _typ([x["ratio"] for x in rows]),
                                        "inside_spread": sum(x["inside_spread"] for x in rows),
                                        "n": len(rows), "tasks": rows}

            # 2b. compute time on the anchor GPU (work / published throughput)
            rows = []
            for t in have:
                at = anchor_time(meas(t), rs[t]["fields"])
                if at:
                    rows.append({"task": t, "gpu": meas(t).get("anchor_gpu") or REFERENCE_GPU,
                                 "measured": _r(at[0], 1), "forecast": _r(at[1], 1),
                                 "ratio": _r(at[1] / at[0]), "status": time_status(at[1], at[0])})
            if rows:
                ms["anchor_time"] = {"centre": _r(_gm([x["ratio"] for x in rows])),
                                     "typical_deviation": _typ([x["ratio"] for x in rows]),
                                     "good": sum(1 for x in rows if x["status"] == "good"),
                                     "n": len(rows), "tasks": rows}

            # 3. VRAM: measured peak, and measured upper bounds
            rows = []
            for t in have:
                m, g = meas(t).get("min_vram_gb"), rs[t]["fields"].get("min_vram_gb")
                if m and g:
                    rows.append({"task": t, "measured": m, "forecast": g, "ratio": _r(g / m),
                                 "status": vram_status(g, m)})
            if rows:
                ms["vram"] = {"centre": _r(_gm([x["ratio"] for x in rows])),
                              "typical_deviation": _typ([x["ratio"] for x in rows]),
                              "below_peak": sum(1 for x in rows if x["forecast"] < x["measured"]),
                              "close": sum(1 for x in rows if scoring.VRAM_CLOSE[0] <= x["ratio"] <= scoring.VRAM_CLOSE[1]),
                              "n": len(rows), "tasks": rows}
            bounds = []
            for t in have:
                b, g = meas(t).get("vram_gb_at_most"), rs[t]["fields"].get("min_vram_gb")
                if b and g is not None:
                    # within the bound but below half of it: the bound is far from
                    # the forecast, so an underestimate is likely (warning)
                    bounds.append({"task": t, "at_most": b, "forecast": g,
                                   "status": "critical" if g > b
                                   else "warning" if g < b * scoring.VRAM_BOUND_FAR else "good"})
            if bounds:
                ms["vram_bound"] = {"above": sum(1 for x in bounds if x["status"] == "critical"),
                                    "n": len(bounds), "tasks": bounds}

            # 4. ladder
            lrows, cells, all_err, uncovered, anchors = [], {}, [], [], {}
            for t in have:
                full = ladder_pairs(meas(t), rs[t]["fields"].get("compute_ratio"))
                if not full:
                    continue
                if meas(t).get("anchor_gpu"):
                    anchors[t] = meas(t)["anchor_gpu"]
                # the anchor (or the reference card) is 1.0 on both sides by
                # definition: kept for the card order and the pick, not scored
                anchor = meas(t).get("anchor_gpu") or REFERENCE_GPU
                cells[t] = {g: {"measured": e, "forecast": _r(f),
                                "status": "anchor" if g == anchor else
                                          ratio_status(f, e) if f is not None else "none"}
                            for g, (e, f) in full.items()}
                # the anchor's own cell shows the time deviation on that card:
                # compute time on the anchor, else the reference-card time
                at = anchor_time(meas(t), rs[t]["fields"])
                rm, rg = meas(t).get("reference_sec"), rs[t]["fields"].get("reference_sec")
                tm = at if at else (rm, rg) if anchor == REFERENCE_GPU and rm and rg else None
                if anchor in cells[t] and tm:
                    cells[t][anchor].update(time_ratio=_r(tm[1] / tm[0]), time_status=time_status(tm[1], tm[0]))
                scored = [g for g in full if g != anchor]
                missing = [g for g in scored if full[g][1] is None]
                if missing:
                    uncovered.append({"task": t, "missing": missing, "measured": len(scored)})
                lp = {g: v for g, v in full.items() if v[1] is not None}
                if len(lp) < 2:
                    continue
                exp, got = {g: v[0] for g, v in lp.items()}, {g: v[1] for g, v in lp.items()}
                err = [abs(math.log(got[g] / exp[g])) for g in lp if g != anchor]
                all_err += err
                pick, best = min(lp, key=lambda g: got[g]), min(lp, key=lambda g: exp[g])
                regret = exp[pick] / exp[best]
                rank = _spearman(list(exp.values()), list(got.values())) if len(lp) >= 3 else None
                lrows.append({"task": t, "gpus": len(lp), "scored_gpus": len(err),
                              "typical_error": _r(math.exp(statistics.median(err))),
                              "p90_error": _r(math.exp(_q(err, 0.9))),
                              "rank_corr": _r(rank, 2) if rank is not None else None, "pick": pick, "fastest": best,
                              "time_regret": _r(regret),
                              "status": "critical" if regret > 1.0001 else
                                        "warning" if rank is not None and rank < 0.7 else "good"})
            if cells:
                ms["ladder"] = {
                    "pairs": len(all_err),
                    "typical_error": _r(math.exp(statistics.median(all_err))) if all_err else None,
                    "p90_error": _r(math.exp(_q(all_err, 0.9))) if all_err else None,
                    "median_rank_corr": _r(statistics.median(ranks), 2) if (ranks := [x["rank_corr"] for x in lrows
                                                                                  if x["rank_corr"] is not None]) else None,
                    "pick_slower": pick_counts(cells)[1],
                    "n": pick_counts(cells)[0], "tasks": lrows, "cells": cells, "uncovered": uncovered,
                    "anchors": anchors}
            svc["measurements"].append(ms)
        out["services"].append(svc)
    return out



def pick_counts(cells: dict) -> tuple[int, int]:
    """(tasks, picks slower than the fastest) over the ladder cells. A task
    counts when it is measured on two or more cards, whatever the service
    forecast: the pick is the card with the smallest forecast ratio, and a
    task with no forecast for its fastest measured card is a slower pick."""
    n = slower = 0
    for c in cells.values():
        if len(c) < 2:
            continue
        n += 1
        best = min(c, key=lambda g: c[g]["measured"])
        lp = [g for g in c if c[g]["forecast"] is not None]
        if not lp or c[min(lp, key=lambda g: c[g]["forecast"])]["measured"] / c[best]["measured"] > 1.0001:
            slower += 1
    return n, slower


def headline_tiles(svc: dict, th: dict) -> list[tuple[str, str, str, str | None]]:
    """The top numbers of a service over all its measurement sets, as
    (name, value, note, status) — one tile per number. Read from a service of
    report.json, so it can be recomputed for any stored report."""
    ms = svc["measurements"]

    def rows(key):
        return [x for m in ms if key in m for x in m[key]["tasks"]]

    out = []
    ft = svc["forecast_time"]
    if ft["median_sec"] is not None:
        out.append(("Forecast time, median", f"{ft['median_sec']} s",
                    f"p90 {ft['p90_sec']} s · max {ft['max_sec']} s", None))
    fm = sum(v["match"] for v in svc["fields"].values())
    ftot = sum(v["total"] for v in svc["fields"].values())
    if ftot:
        share = fm / ftot
        good, warn = th.get("fields_good", scoring.FIELDS_GOOD), th.get("fields_warning", scoring.FIELDS_WARN)
        out.append(("Classification fields", f"{fm}/{ftot}", f"match the description · {share:.1%}",
                    "good" if share >= good else "warning" if share >= warn else "critical"))
    rt = rows("reference_time")
    if rt:
        typ = _typ([x["ratio"] for x in rt])
        out.append(("Time, typical deviation", _x(typ),
                    f"in spread {sum(x['inside_spread'] for x in rt)}/{len(rt)} · reference card",
                    _band(typ, th["time_good"], th["time_warning"])))
    at = rows("anchor_time")
    if at:
        ratios = [x["ratio"] for x in at]
        typ = _typ(ratios)
        out.append(("Compute time, typical deviation", _x(typ),
                    f"centre {_x(_r(_gm(ratios)))} · {len(at)} tasks",
                    _band(typ, th["time_good"], th["time_warning"])))
    vr = rows("vram")
    if vr:
        below = sum(1 for x in vr if x["forecast"] < x["measured"])
        out.append(("VRAM below the measured peak", f"{below}/{len(vr)}",
                    f"centre {_x(_r(_gm([x['ratio'] for x in vr])))}", "good" if below == 0 else "critical"))
        lo, hi = scoring.VRAM_CLOSE
        close = sum(1 for x in vr if lo <= x["ratio"] <= hi)
        out.append(("VRAM close to the measured peak", f"{close}/{len(vr)}",
                    f"forecast x{lo:.2f}-{hi:.2f} of the peak · higher is better",
                    "good" if close == len(vr) else "warning" if 2 * close >= len(vr) else "critical"))
    vb = rows("vram_bound")
    if vb:
        above = sum(1 for x in vb if x["status"] == "critical")
        out.append(("VRAM above a measured bound", f"{above}/{len(vb)}", "an upper bound does not check an underestimate",
                    "good" if above == 0 else "critical"))
    lds = [m["ladder"] for m in ms if "ladder" in m]
    err, n, slower, miss, tot = [], 0, 0, 0, 0
    for ld in lds:
        # from the cells, so reports stored with the old count read the same way
        ld_n, ld_slower = pick_counts(ld["cells"])
        n += ld_n
        slower += ld_slower
        miss += sum(len(x["missing"]) for x in ld.get("uncovered", []))
        for t, cells in ld["cells"].items():
            anchor = ld.get("anchors", {}).get(t) or REFERENCE_GPU
            tot += sum(1 for g in cells if g != anchor)
            lp = {g: c for g, c in cells.items() if c["forecast"] is not None}
            if len(lp) >= 2:
                err += [abs(math.log(c["forecast"] / c["measured"])) for g, c in lp.items() if g != anchor]
    if err:
        typ = _r(math.exp(statistics.median(err)))
        out.append(("Ladder, typical error", _x(typ),
                    f"p90 {_x(_r(math.exp(_q(err, 0.9))))} · {len(err)} pairs",
                    _band(typ, th["ratio_good"], th["ratio_warning"])))
    if n:
        out.append(("Fastest GPU picked right", f"{n - slower}/{n}", "",
                    "good" if slower == 0 else "critical"))
    if miss:
        out.append(("Measured GPUs with no forecast", f"{miss}/{tot}", "", "critical"))
    return out


# --------------------------------------------------------------------- markdown

def _md_ratio(x: dict, unit: str, extra: str) -> list[str]:
    return ([f"centre x{x['centre']}, typical deviation x{x['typical_deviation']}, {extra}", "",
             f"| task | measured, {unit} | forecast, {unit} | forecast / measured | status |", "|---|---|---|---|---|"]
            + [f"| {r['task']} | {r['measured']} | {r['forecast']} | x{r['ratio']} | {r['status']} |" for r in x["tasks"]]
            + [""])


def _md_time(m: dict) -> list[str]:
    L = []
    if "anchor_time" in m:
        x = m["anchor_time"]
        L += ["##### Compute time on the anchor GPU", "",
              f"centre x{x['centre']}, typical deviation x{x['typical_deviation']}, "
              f"within x{TIME_GOOD}: {x['good']}/{x['n']}", "",
              "| task | GPU | measured, s | forecast, s | forecast / measured | status |", "|---|---|---|---|---|---|"]
        L += [f"| {r['task']} | {r['gpu']} | {r['measured']} | {r['forecast']} | x{r['ratio']} | {r['status']} |"
              for r in x["tasks"]]
        L.append("")
    if "reference_time" in m:
        x = m["reference_time"]
        L += ["##### Reference-card time", ""] + _md_ratio(x, "s", f"inside the issued spread {x['inside_spread']}/{x['n']}")
    return L


def _md_vram(m: dict) -> list[str]:
    L = []
    if "vram" in m:
        x = m["vram"]
        close = f", close to it (x{scoring.VRAM_CLOSE[0]:.2f}-{scoring.VRAM_CLOSE[1]:.2f}) {x['close']}/{x['n']}" if "close" in x else ""
        L += ["##### Measured peak", ""] + _md_ratio(x, "GB", f"below the measured peak {x['below_peak']}/{x['n']}{close}")
    if "vram_bound" in m:
        vb = m["vram_bound"]
        L += ["##### Upper bound", "", f"forecast above the measured bound in {vb['above']}/{vb['n']} tasks"
              + "".join(f"; {x['task']}: {x['forecast']} GB > {x['at_most']} GB"
                        for x in vb["tasks"] if x["status"] == "critical"), ""]
    return L


def _md_ladder(m: dict) -> list[str]:
    L, ld = [], m.get("ladder")
    if not ld:
        return L
    if ld["tasks"]:
        L += ["##### compute_ratio", "",
              f"{ld['pairs']} (task, GPU) pairs: typical error x{ld['typical_error']}, p90 x{ld['p90_error']}; "
              f"median rank correlation {ld['median_rank_corr']}; "
              f"pick slower than the fastest in {ld['pick_slower']}/{ld['n']} tasks", "",
              "| task | GPUs | typical error | p90 | rank corr | picked | fastest | time regret | status |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {x['task']} | {x['gpus']} | x{x['typical_error']} | x{x['p90_error']} | {x['rank_corr']} | "
              f"{x['pick']} | {x['fastest']} | x{x['time_regret']} | {x['status']} |" for x in ld["tasks"]]
        L.append("")
    if ld["uncovered"]:
        L += ["##### Coverage", "", "measured GPUs without a forecast: " + "; ".join(
            f"{x['task']} {len(x['missing'])}/{x['measured']}" for x in ld["uncovered"]), ""]
    return L


def render_md(rep: dict) -> str:
    L = []
    for s in rep["services"]:
        L += [f"## {s['service']}", "", f"{len(s['tasks'])} tasks; versions: " + "; ".join(s["versions"]), ""]
        ft = s["forecast_time"]
        if ft["median_sec"] is not None:
            L += ["### Forecast time", "", f"median {ft['median_sec']} s, p90 {ft['p90_sec']} s, max {ft['max_sec']} s", ""]
        L += ["### Classification / assay fields", "", "| field | match | differs in |", "|---|---|---|"]
        for f, v in s["fields"].items():
            bad = [t for t, x in v["tasks"].items() if x["status"] != "good"]
            L.append(f"| {f} | {v['match']}/{v['total']} | {', '.join(bad) or '-'} |")
        L.append("")
        for title, part in (("Time", _md_time), ("VRAM", _md_vram), ("Ladder", _md_ladder)):
            blocks = [(m, part(m)) for m in s["measurements"]]
            blocks = [(m, b) for m, b in blocks if b]
            if blocks:
                L += [f"### {title}", ""]
                for m, b in blocks:
                    L += [f"#### {m['id']} (independent: {', '.join(map(str, m['independent']))}; "
                          f"{len(m['tasks'])} tasks)", ""] + b
    return "\n".join(L)


# --------------------------------------------------------------------- html

_ICON = {"good": "✓", "warning": "!", "critical": "✗", "none": "–", "anchor": "◦"}
_WORD = {"good": "accurate", "warning": "miss", "critical": "problem", "none": "no forecast",
         "anchor": "anchor (1.00 by definition, not scored)"}

# Status colours: the reference palette's fixed status set (good / warning /
# critical), always paired with an icon and a word — never colour alone.
_CSS = """
.viz-root{color-scheme:light;--surface-0:#f3f3f1;--surface-1:#fcfcfb;--border:#dedcd6;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#6f6d67;
--good:#0ca30c;--warning-ink:#8a5d00;--critical:#d03b3b;
--good-bg:rgba(12,163,12,.13);--warning-bg:rgba(250,178,25,.24);--critical-bg:rgba(208,59,59,.15);--none-bg:rgba(119,117,111,.10)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9a988f;
--warning-ink:#fab219;--good-bg:rgba(12,163,12,.24);--warning-bg:rgba(250,178,25,.20);--critical-bg:rgba(208,59,59,.28);--none-bg:rgba(195,194,183,.10)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;
--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9a988f;
--warning-ink:#fab219;--good-bg:rgba(12,163,12,.24);--warning-bg:rgba(250,178,25,.20);--critical-bg:rgba(208,59,59,.28);--none-bg:rgba(195,194,183,.10)}
html,body{margin:0;background:#f3f3f1}
@media (prefers-color-scheme:dark){html:where(:not([data-theme="light"])),html:where(:not([data-theme="light"])) body{background:#121211}}
html[data-theme="dark"],html[data-theme="dark"] body{background:#121211}
.viz-root{background:var(--surface-0);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:24px 16px 48px;min-height:100vh;box-sizing:border-box}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px} h3{font-size:16px;margin:32px 0 6px} h4{font-size:14px;margin:20px 0 6px}
.sub{color:var(--text-secondary);margin:0 0 12px} .muted{color:var(--text-muted);font-weight:400}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin:12px 0 4px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--text-secondary);font-size:12px} .tile .v{font-size:24px;font-weight:600;margin:2px 0}
.tile .n{color:var(--text-muted);font-size:12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.panel{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--text-secondary);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--surface-1)}
tr:last-child td{border-bottom:0} td.num{text-align:right}
.st{display:inline-flex;align-items:center;gap:5px;padding:1px 8px;border-radius:999px;font-size:12px;color:var(--text-primary)}
.st.good{background:var(--good-bg)} .st.warning{background:var(--warning-bg)} .st.critical{background:var(--critical-bg)} .st.none{background:var(--none-bg)}
.st b{font-weight:700} .st.good b{color:var(--good)} .st.warning b{color:var(--warning-ink)} .st.critical b{color:var(--critical)} .st.none b{color:var(--text-muted)}
td.cell{text-align:center;font-size:12px;min-width:56px;padding:7px 8px}
td.cell.good{background:var(--good-bg)} td.cell.warning{background:var(--warning-bg)}
td.cell.critical{background:var(--critical-bg)} td.cell.none{background:var(--none-bg);color:var(--text-muted)}
td.cell.anchor{color:var(--text-muted)} .st.anchor{background:var(--none-bg)} .st.anchor b{color:var(--text-muted)}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 0}
.note{color:var(--text-secondary);font-size:13px;margin:6px 0}
.toggle{float:right;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);border-radius:8px;padding:4px 10px;cursor:pointer;font:inherit}
"""


def _st(status: str, text: str | None = None) -> str:
    return f'<span class="st {status}"><b>{_ICON[status]}</b>{html.escape(text or _WORD[status])}</span>'


def _tile(k: str, v: str, n: str = "", status: str | None = None) -> str:
    badge = _st(status) if status else ""
    return (f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
            f'<div class="n">{badge}<span>{html.escape(n)}</span></div></div>')


def _tbl(head: list[str], rows: list[list[str]], num: frozenset = frozenset()) -> str:
    h = "".join(f'<th class="{"num" if i in num else ""}">{html.escape(c)}</th>' for i, c in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f'<div class="panel"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _x(v) -> str:
    return "–" if v is None else f"x{v:.2f}"


def _band(v: float, good: float, warn: float) -> str:
    return "good" if v <= good else "warning" if v <= warn else "critical"


def _ind(flags: list[bool]) -> str:
    return "independent" if flags == [True] else "not independent — service calibration data" if flags == [False] else "mixed"


VRAM_NOTE = ('VRAM, GB: the measured peak, or an upper bound (the run completed on a card of that size), '
             'against the forecast requirement. An upper bound catches over-provisioning beyond the card only; '
             'it does not check an underestimate.')
LADDER_NOTE = ('Ladder per GPU: forecast / measured compute_ratio (1.00 = exact; hover for both '
               'values). An empty cell is a GPU not measured for that task; "vs" names the anchor GPU a '
               'published measurement is relative to. In the anchor\'s own cell (column marked ◦ anchor): '
               'forecast / measured time on that card (compute time on the anchor, else the reference-card '
               'time); it does not enter the ladder errors.')


def _html_time(m: dict, th: dict) -> list[str]:
    P, e = [], html.escape
    if "reference_time" in m:
        rt = m["reference_time"]
        P.append(f'<p class="note">Time on the reference card: centre {_x(rt["centre"])}, typical deviation '
                 f'{_x(rt["typical_deviation"])}, inside the issued spread {rt["inside_spread"]}/{rt["n"]}.</p>')
        P.append(_tbl(["task", "measured, s", "forecast, s", "forecast / measured", "status",
                       "issued spread", "measured inside it"],
                      [[e(x["task"]), str(x["measured"]), str(x["forecast"]), _x(x["ratio"]), _st(x["status"]),
                        f"x{x['spread_factor']:.3g}",
                        _st("good", "yes") if x["inside_spread"] else _st("critical", "no")]
                       for x in rt["tasks"]], frozenset({1, 2, 3, 5})))
    if "anchor_time" in m:
        at = m["anchor_time"]
        P.append(f'<p class="note">Compute time on the anchor GPU (measured: the work of the task / the '
                 f'published throughput; forecast: compute time on the reference card x the forecast ratio '
                 f'of the anchor): centre {_x(at["centre"])}, typical deviation '
                 f'{_x(at["typical_deviation"])}, within x{th["time_good"]}: {at["good"]}/{at["n"]}.</p>')
        P.append(_tbl(["task", "GPU", "measured, s", "forecast, s", "forecast / measured", "status"],
                      [[e(x["task"]), e(x["gpu"]), str(x["measured"]), str(x["forecast"]), _x(x["ratio"]),
                        _st(x["status"])] for x in at["tasks"]], frozenset({2, 3, 4})))
    return P


def _html_vram(m: dict, th: dict) -> list[str]:
    P, e = [], html.escape
    if "vram" in m or "vram_bound" in m:
        rows = [[e(x["task"]), str(x["measured"]), str(x["forecast"]), _x(x["ratio"]),
                 _st(x["status"], "below the peak" if x["status"] == "critical"
                     else "over-provisioned" if x["status"] == "warning" else None)]
                for x in m.get("vram", {}).get("tasks", [])]
        rows += [[e(x["task"]), f"≤ {x['at_most']}", str(x["forecast"]), "–",
                  _st(x["status"], "above the bound" if x["status"] == "critical"
                      else "far below the bound" if x["status"] == "warning" else "within the bound")]
                 for x in m.get("vram_bound", {}).get("tasks", [])]
        P.append(_tbl(["task", "measured", "forecast", "forecast / measured", "status"], rows, frozenset({1, 2, 3})))
    return P


def _html_ladder(m: dict, th: dict) -> list[str]:
    P, e = [], html.escape
    if "ladder" in m:
        ld = m["ladder"]
        if ld["tasks"]:
            P.append(f'<p class="note">Ladder: {ld["pairs"]} (task, GPU) pairs, typical error {_x(ld["typical_error"])}, '
                     f'p90 {_x(ld["p90_error"])}, median rank correlation {ld["median_rank_corr"]}.</p>')
            P.append(_tbl(["task", "GPUs", "typical error", "p90", "rank corr", "picked as fastest",
                           "really fastest", "time regret", "status"],
                          [[e(x["task"]), str(x["gpus"]), _x(x["typical_error"]), _x(x["p90_error"]),
                            str(x["rank_corr"]), e(x["pick"]), e(x["fastest"]), _x(x["time_regret"]),
                            _st(x["status"], "slower pick" if x["status"] == "critical"
                                else "order off" if x["status"] == "warning" else "right pick")]
                           for x in ld["tasks"]], frozenset({1, 2, 3, 4, 7})))
        gpus = sorted({g for c in ld["cells"].values() for g in c})
        body = ""
        for t, c in ld["cells"].items():
            tds = []
            for g in gpus:
                x = c.get(g)
                if x is None:
                    tds.append("<td></td>")
                elif x["status"] == "anchor":
                    if x.get("time_ratio") is None:
                        tds.append(f'<td class="cell none" title="{e(g)}: anchor, no time measured">– none</td>')
                    else:
                        st = x["time_status"]
                        tip = f"{g}: anchor · time forecast / measured"
                        tds.append(f'<td class="cell {st}" title="{e(tip)}">{_ICON[st]} {x["time_ratio"]:.2f}</td>')
                elif x["forecast"] is None:
                    tip = f"{g}: measured {x['measured']} · no forecast"
                    tds.append(f'<td class="cell none" title="{e(tip)}">– none</td>')
                else:
                    r = x["forecast"] / x["measured"]
                    tip = f"{g}: measured {x['measured']} · forecast {x['forecast']}"
                    tds.append(f'<td class="cell {x["status"]}" title="{e(tip)}">{_ICON[x["status"]]} {r:.2f}</td>')
            anchor = ld["anchors"].get(t)
            label = e(t) + (f' <span class="muted">vs {e(anchor)}</span>' if anchor else "")
            body += f"<tr><td>{label}</td>{''.join(tds)}</tr>"
        anchor_gpus = {ld["anchors"].get(t) or REFERENCE_GPU for t in ld["cells"]}
        head = "".join(f"<th>{e(g)}{'<br>◦ anchor' if g in anchor_gpus else ''}</th>" for g in gpus)
        P.append(f'<div class="panel"><table><thead><tr><th>task</th>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        P.append('<div class="legend">' + " ".join(_st(k) for k in ("good", "warning", "critical", "none")) + "</div>")
        if ld["uncovered"]:
            P.append('<p class="note">' + _st("critical", "no forecast")
                     + " measured GPUs the service listed no row for (e.g. filtered out by its VRAM requirement): "
                     + "; ".join(f"{e(x['task'])} {len(x['missing'])}/{x['measured']}" for x in ld["uncovered"]) + "</p>")
    return P


# Rows of the header tiles: the base of the forecast, then what is derived from it
# (the forecast's own speed leads the first row).
_TILE_ROWS = (
    ("Base: classification and VRAM", ("Forecast time, median", "Classification fields", "VRAM below the measured peak",
                                       "VRAM close to the measured peak", "VRAM above a measured bound")),
    ("Derived: time, performance and ladder", ("Time, typical deviation", "Compute time, typical deviation",
                                               "Ladder, typical error", "Fastest GPU picked right",
                                               "Measured GPUs with no forecast")),
    ("Other", ()),
)


def render_html(rep: dict) -> str:
    th, e = rep["thresholds"], html.escape
    P = ["<h1>Forecast quality report</h1>",
         f'<p class="sub">Generated {e(rep["created_utc"])}. Each value is marked '
         f'{_st("good")} time within x{th["time_good"]}, ladder ratio within '
         f'x{th["ratio_good"]}, VRAM up to x{th["vram_over_warning"]} above the measured peak · '
         f'{_st("warning")} time within x{th["time_warning"]}, ratio within x{th["ratio_warning"]}, VRAM over-provisioned'
         + (f' or below {th["vram_bound_far"]:.0%} of an upper bound' if "vram_bound_far" in th else "")
         + (f', a numeric field off by less than {th["field_near"]:.0%}' if "field_near" in th else "")
         + ' · '
         f'{_st("critical")} beyond that, VRAM below the peak (OOM risk) or above a measured bound, '
         f'a field that differs, a slower GPU picked as the fastest.</p>']
    for s in rep["services"]:
        P.append("<h2>Summary — all measurement sets</h2>")
        vers = [" · ".join(f"{k} {v}" for k, v in json.loads(v).items()) for v in s["versions"]]
        P.append(f'<p class="sub">service {e(s["service"])} · {len(s["tasks"])} tasks · '
                 + " | ".join(e(v) for v in vers) + "</p>")
        # Base first (classification and VRAM), then what is derived from it
        # (time, ladder); unknown names go last.
        tiles = headline_tiles(s, th)
        for label, names in _TILE_ROWS:
            row = [t for t in tiles if t[0] in names] if names else \
                [t for t in tiles if not any(t[0] in n for _, n in _TILE_ROWS if n)]
            row.sort(key=lambda t: names.index(t[0]) if names else 0)
            if row:
                P.append(f'<p class="note">{e(label)}</p>')
                P.append('<div class="tiles">' + "".join(_tile(*t) for t in row) + "</div>")

        # classification: tasks x fields
        fnames = list(s["fields"])
        rows = []
        for t in s["tasks"]:
            row = [e(t)]
            for f in fnames:
                x = s["fields"][f]["tasks"].get(t)
                if x is None:
                    row.append('<span class="muted">–</span>')
                    continue
                tip = f"expected {x['expected']} · forecast {x['forecast']}"
                shown = ("fp32" if f == "precision" else "not read") if x["forecast"] is None else x["forecast"]
                txt = str(shown) if x["status"] == "good" else f"{shown} ≠ {x['expected']}"
                row.append(f'<span title="{e(tip)}">{_st(x["status"], txt)}</span>')
            rows.append(row)
        P.append("<h3>Classification / assay fields</h3>")
        P.append('<p class="note">Expected values are read from the task code and stored in the task description. '
                 'Hover a cell for expected / forecast.</p>')
        P.append(_tbl(["task"] + fnames, rows))

        notes = {"VRAM": VRAM_NOTE, "Ladder": LADDER_NOTE}
        for title, part in (("Time", _html_time), ("VRAM", _html_vram), ("Ladder", _html_ladder)):
            blocks = [(m, part(m, th)) for m in s["measurements"]]
            blocks = [(m, b) for m, b in blocks if b]
            if not blocks:
                continue
            P.append(f"<h3>{title}</h3>")
            if title in notes:
                P.append(f'<p class="note">{notes[title]}</p>')
            for m, b in blocks:
                P.append(f"<h4>{e(m['id'])} <span class='muted'>· {_ind(m['independent'])} · "
                         f"{len(m['tasks'])} tasks</span></h4>")
                P += b
    toggle = ('<button class="toggle" type="button" onclick="var r=document.documentElement;'
              "var d=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;"
              "r.dataset.theme=d?'light':'dark'\">Light / dark</button>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Forecast quality report</title><style>{_CSS}</style></head>"
            f'<body><div class="viz-root"><div class="wrap">{toggle}' + "\n".join(P) + "</div></div></body></html>\n")


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service")
    ap.add_argument("--version", help="only results of this service version (e.g. a calibration_id)")
    ap.add_argument("--measurement", help="only this measurement set of the descriptions (default: each)")
    ap.add_argument("--out", default=str(ROOT / "results" / "report"), help="output directory")
    args = ap.parse_args()
    tasks = ROOT / "tasks"
    descs = {str(p.relative_to(tasks).with_suffix("")): json.loads(p.read_text()) for p in tasks.rglob("*.json")}
    rep = analyze(latest(args.service, args.version), descs, args.measurement)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(render_md(rep) + "\n")
    (out / "report.html").write_text(render_html(rep))
    print(f"{out}: report.json, report.md, report.html")


if __name__ == "__main__":
    main()
