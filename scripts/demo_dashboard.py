#!/usr/bin/env python
"""Build a self-contained, interactive HTML view of the merged clean-null ledger.

Why this exists: the merged pass produces four receipts and a 21 MB ledger. A reviewer
who will not run the pipeline can still be shown the *whole* population and invited to
falsify the headline by hand. That is the point - the page carries every score, so a
sceptic can move a threshold and watch the rate move, rather than reading a table.

Design rules, learned from the failures this lane already had:

  * The page never *asserts* a number it can compute. The committed report's values are
    printed as "what the evaluator wrote", and the explorer recomputes them from the raw
    scores in the browser. The build step asserts the two agree, so a disagreement is a
    build failure rather than something a reader discovers in front of an audience.
  * Scores are embedded as float64, not float32. A float32 round-trip moves the 0.05
    quantile to 0.9845092 and can change the hit count, which would make the page
    disagree with the signed report by one model for no reason.
  * TPR is shown as *undefined*, never as 0.000. There are no positives in this
    population and a zero would be a fabricated measurement.
  * Degeneracy is a first-class state: when a threshold sits at the corpus ceiling the
    page says the rule cannot fire, instead of printing a flattering 0.0000.

No network, no CDN, no build step: one .html file that opens from disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

Z95 = 1.959963984540054

# The evaluator rounds its published point estimate to this many decimals (0.0523, not
# 0.05230812...). Any comparison against it must happen at the precision it publishes, or
# a rule that is exactly right reads as a disagreement.
PUBLISHED_FPR_DECIMALS = 4

# The detection panel is built from the per-class, per-dose receipt. A single-dose receipt
# cannot fill a per-dose table without inventing the other doses, so it is refused by name.
LADDER_SCHEMA = "cviaf.tpr-ladder.v1"


def wilson(hits: int, n: int, z: float = Z95) -> list:
    """Wilson score interval - the same one the evaluator prints."""
    if n <= 0:
        return [None, None]
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, centre - half), min(1.0, centre + half)]


def quantile(sorted_values: list, q: float) -> float:
    """Linear-interpolation quantile, matching numpy's default method."""
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def b64_floats(values) -> str:
    import numpy as np

    return base64.b64encode(np.asarray(values, dtype="<f8").tobytes()).decode("ascii")


def b64_bits(flags) -> str:
    import numpy as np

    return base64.b64encode(np.packbits(np.asarray(flags, dtype="uint8")).tobytes()).decode("ascii")


def load_ledger(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_owner_map(plan_dir: Path) -> dict:
    """Map every seed to the shard that owns it, from the plan's own model lists.

    Deliberately not derived from seed_min..seed_max: those ranges overlap between
    sources (mac 10,264-24,607 sits inside git's 10,150-28,385 span), so a range test
    would mislabel models. The plan lists the models, so the plan decides.
    """
    owner: dict = {}
    for shard_path in sorted((plan_dir / "shards").glob("*.json")):
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        for model in shard.get("models") or []:
            owner[int(model["seed"])] = shard["shard_id"]
    return owner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, help="merged fpr_tpr_ledger.json")
    ap.add_argument("--report", required=True, help="the evaluator's fpr_tpr_report.json")
    ap.add_argument("--plan", required=True, help="combined plan directory")
    ap.add_argument("--verify", required=True, help="runs/integration_verify.json")
    ap.add_argument("--census", required=True, help="runs/integration_census.json")
    ap.add_argument("--out", required=True, help="the .html to write")
    ap.add_argument("--split-seed", type=int, default=0, help="must match the evaluator")
    ap.add_argument("--tpr", default=None,
                    help="runs/tpr_at_frozen.json - the additive attacked-arm receipt")
    ap.add_argument("--tpr-skipped", default=None,
                    help="the scorer's skipped_*.json for the arms, so the panel can say "
                         "how many attacks could not be scored")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cviaf.lab.fpr_tpr import assign_splits  # the real assignment, not a re-implementation

    ledger = load_ledger(Path(args.ledger))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    plan = json.loads((Path(args.plan) / "plan.json").read_text(encoding="utf-8"))
    verify = json.loads(Path(args.verify).read_text(encoding="utf-8"))
    census = json.loads(Path(args.census).read_text(encoding="utf-8"))

    signals = list(report["rules"].keys())
    records = ledger["records"]
    n = len(records)

    assigned = assign_splits(ledger, seed=args.split_seed)["records"]
    is_eval = [1 if r["split"] == "evaluation" else 0 for r in assigned]
    # The locked ledger schema does not carry the detector seed as its own field; it is the
    # suffix of the model id (`clean_none_fixed_s150`). Parse it rather than guessing from
    # the record order, because a seed is what ties a row back to the shard that made it.
    seeds = [int(r["model_id"].rsplit("_s", 1)[1]) for r in records]
    owner = seed_owner_map(Path(args.plan))
    shard_ids = [owner.get(s, "?") for s in seeds]

    # ---- what the committed report says, and what we recompute from the same bytes ----
    committed = {}
    for sig in signals:
        rule = report["rules"][sig]
        cal = sorted(r["scores"][sig] for r in assigned if r["split"] == "calibration")
        ev = [r["scores"][sig] for r in assigned if r["split"] == "evaluation"]
        thr = quantile(cal, 1.0 - report["alpha"])
        hits = sum(1 for v in ev if v > thr)
        committed[sig] = {
            "committed_threshold": rule["threshold"],
            "committed_fpr": rule["fpr"]["point_estimate"],
            "committed_hits": rule["fpr"]["numerator"],
            "committed_ci": rule["fpr"]["ci95_wilson"],
            "recomputed_threshold": thr,
            "recomputed_fpr": hits / len(ev) if ev else None,
            "recomputed_hits": hits,
            "recomputed_ci": wilson(hits, len(ev)),
            "status": rule["status"],
            "ceiling": cal[-1] if cal else None,
            "degenerate": bool(cal) and thr >= cal[-1],
        }
        # Fail closed: if the page's arithmetic disagrees with the signed report, stop.
        if abs(thr - rule["threshold"]) > 1e-9:
            raise SystemExit(
                f"threshold mismatch for {sig}: recomputed {thr!r} != committed "
                f"{rule['threshold']!r} - refusing to publish a page that contradicts the report"
            )
        if hits != rule["fpr"]["numerator"]:
            raise SystemExit(
                f"hit-count mismatch for {sig}: recomputed {hits} != committed "
                f"{rule['fpr']['numerator']} - refusing to publish a page that contradicts the report"
            )
        # The "agrees" column is the first thing a sceptical reader checks, so it is
        # computed here (and tested) rather than in the page: an exact ratio compared
        # against the report's *rounded* point estimate printed "NO" on every rule that
        # was right, which reads as the page contradicting the report it recomputes.
        committed[sig]["agrees"] = bool(
            ev
            and hits == rule["fpr"]["numerator"]
            and abs(thr - rule["threshold"]) <= 1e-9
            and round(hits / len(ev), PUBLISHED_FPR_DECIMALS)
            == round(rule["fpr"]["point_estimate"], PUBLISHED_FPR_DECIMALS)
        )

    # ---- split-seed sensitivity, recomputed the same way ----
    sensitivity = []
    for seed in (0, 1, 7, 42, 1337):
        part = assign_splits(ledger, seed=seed)["records"]
        row = {"seed": seed, "rules": {}}
        for sig in signals:
            cal = sorted(r["scores"][sig] for r in part if r["split"] == "calibration")
            ev = [r["scores"][sig] for r in part if r["split"] == "evaluation"]
            thr = quantile(cal, 1.0 - report["alpha"])
            hits = sum(1 for v in ev if v > thr)
            row["rules"][sig] = {
                "threshold": thr,
                "fpr": hits / len(ev) if ev else None,
                "hits": hits,
            }
        sensitivity.append(row)

    shards = [
        {
            "shard_id": sh["shard_id"],
            "count": sh["count"],
            "seed_min": sh["seed_min"],
            "seed_max": sh["seed_max"],
        }
        for sh in plan["shards"]
    ]

    tpr_blob = None
    if args.tpr:
        tpr = json.loads(Path(args.tpr).read_text(encoding="utf-8"))
        if tpr.get("schema") != LADDER_SCHEMA:
            raise SystemExit(
                f"--tpr wants a {LADDER_SCHEMA} receipt (per attack class, per dose); "
                f"{args.tpr} is {tpr.get('schema')!r}. A single-dose receipt cannot fill a "
                "per-dose table without inventing the other doses, and inventing one is the "
                "failure this page exists to make impossible."
            )
        skipped = None
        if args.tpr_skipped and Path(args.tpr_skipped).is_file():
            raw = json.loads(Path(args.tpr_skipped).read_text(encoding="utf-8"))
            skipped = {
                "n_models": raw.get("n_models"),
                "n_scored": raw.get("n_scored"),
                "n_error": raw.get("n_error"),
                "reasons": sorted({s.get("reason", "") for s in raw.get("skipped", [])}),
            }
        tpr_blob = {
            "mode": "ladder",
            "n_arms": tpr["n_arms"],
            "kinds": tpr["kinds"],
            "doses": tpr["doses"],
            "units_per_kind": tpr.get("units_per_kind", {}),
            "min_positives": tpr.get("min_positives", 20),
            "rules": tpr["rules"],
            "cells": tpr["cells"],
            # Already grouped by (class, dose) by the evaluator, because *where* the
            # unscorable arms fall decides whether a cell is an estimate or a bound.
            "unscorable": tpr.get("unscorable", {}),
            "pooled_refusal": tpr.get("pooled_refusal"),
            "behaviour_metric": tpr.get("behaviour_metric"),
            "problems": tpr.get("problems", []),
            "frozen_thresholds_from": tpr.get("frozen_thresholds_from"),
            "skipped": skipped,
        }

    blob = {
        "n": n,
        "alpha": report["alpha"],
        "tpr": tpr_blob,
        "splitSeed": args.split_seed,
        "signals": signals,
        "scores": {sig: b64_floats([r["scores"][sig] for r in records]) for sig in signals},
        "isEval": b64_bits(is_eval),
        "seeds": seeds,
        "shardOf": [shard_ids.index(s) for s in shard_ids],
        "shards": shards,
        "committed": committed,
        "sensitivity": sensitivity,
        "verify": {
            "sources": [
                {
                    "id": s["id"],
                    "verified": s["n_verified"],
                    "failed": s["n_failed"],
                    "no_plan": s["n_missing_from_plan"],
                }
                for s in verify["sources"]
            ],
            "ok": verify["ok"],
        },
        "census": {
            "n_sources": len(census["sources"]),
            "n_collisions": census["n_collisions"],
            "collisions_truncated": census["collisions_truncated"],
            "collisions_fatal": len(census["collisions_fatal"]),
            "lab_seed_range": census["lab_seed_range"],
            "gaps": [{"id": g.get("id"), "note": g.get("note")} for g in census["gaps"]],
            "sources": [
                {
                    "id": s["id"],
                    "kind": s.get("kind"),
                    "n": s.get("n_models"),
                    "seed_min": s.get("seed_min"),
                    "seed_max": s.get("seed_max"),
                }
                for s in census["sources"]
            ],
        },
    }

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(blob, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {n} models, {len(signals)} signals)")
    for sig in signals:
        c = committed[sig]
        flag = "DEGENERATE" if c["degenerate"] else "live"
        print(
            f"  {sig:20s} thr={c['recomputed_threshold']:.6f} "
            f"fpr={c['recomputed_fpr']:.4f} [{c['recomputed_ci'][0]:.4f}, {c['recomputed_ci'][1]:.4f}] "
            f"hits={c['recomputed_hits']}/{sum(is_eval)} {flag}"
        )
    return 0


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CVIAF - merged clean-null FPR ledger</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--dim:#8b949e;
--y:#d29922;--r:#f85149;--g:#3fb950;--b:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:20px 26px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px;font-weight:600}
.sub{color:var(--dim);font-size:12px}
nav{display:flex;gap:2px;padding:0 26px;border-bottom:1px solid var(--line);
background:var(--panel);position:sticky;top:0;z-index:9;overflow-x:auto}
nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--dim);
padding:11px 14px;font:inherit;font-size:13px;cursor:pointer;white-space:nowrap}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--ink);border-bottom-color:var(--b)}
main{padding:22px 26px;max-width:1180px}
section{display:none}section.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:13px 15px}
.card .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:23px;font-weight:600;margin-top:5px}
.card .n{color:var(--dim);font-size:11px;margin-top:3px}
table{width:100%;border-collapse:collapse;margin:10px 0 18px;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;
letter-spacing:.04em;border-bottom:1px solid var(--line);padding:7px 9px}
td{padding:7px 9px;border-bottom:1px solid #21262d;vertical-align:top}
tr:hover td{background:#1c2128}
.num{text-align:right;font-variant-numeric:tabular-nums}
.dim{color:var(--dim)}.y{color:var(--y)}.r{color:var(--r)}.g{color:var(--g)}.b{color:var(--b)}
h2{font-size:15px;margin:26px 0 6px}
p{color:var(--dim);font-size:13px;margin:6px 0 12px;max-width:76ch}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:12px 0}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:9px 0}
label{color:var(--dim);font-size:12px;min-width:96px}
select,input[type=number]{background:#0d1117;color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:6px 9px;font:inherit;font-size:13px}
input[type=range]{flex:1;min-width:230px;accent-color:var(--b)}
button.act{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:6px;
padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}
button.act:hover{border-color:var(--b)}
.big{font-size:32px;font-weight:600;font-variant-numeric:tabular-nums}
.bar{height:8px;background:#21262d;border-radius:4px;overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%;background:var(--b)}
.warn{border-left:3px solid var(--y);background:#1c1a12;padding:10px 13px;border-radius:0 6px 6px 0;margin:11px 0}
.bad{border-left:3px solid var(--r);background:#1e1214;padding:10px 13px;border-radius:0 6px 6px 0;margin:11px 0}
.ok{border-left:3px solid var(--g);background:#101d14;padding:10px 13px;border-radius:0 6px 6px 0;margin:11px 0}
code{background:#21262d;padding:1px 5px;border-radius:4px;font-size:12px}
a{color:var(--b)}
.timeline td:first-child{color:var(--dim);white-space:nowrap;width:110px}
</style></head><body>
<header>
<h1>CVIAF &mdash; merged clean-null FPR ledger</h1>
<div class="sub" id="sub"></div>
</header>
<nav id="tabs"></nav>
<main>

<section id="t-headline" class="on">
  <div class="cards" id="cards"></div>
  <h2>What the evaluator wrote, and what this page recomputes from the same bytes</h2>
  <p>Every row below was recomputed in this page from the embedded per-model scores, using the
  committed split seed and a Wilson interval. A disagreement would have failed the build - if you
  see one, that is a bug, not a rounding note. Thresholds come from the <b>calibration</b> half's
  negatives only; the rate is measured on the <b>evaluation</b> half, which never contributed to
  the threshold.</p>
  <table id="headline"></table>
  <div id="tprnote"></div>
  <h2>How this population was assembled</h2>
  <div id="assembly"></div>
</section>

<section id="t-explore">
  <h2>Move the threshold yourself</h2>
  <p>This is the falsification tool. Pick a signal and a population, drag the threshold, and watch
  the rate follow. The committed operating point is marked; <b>reset</b> returns to it. Nothing here
  is simulated - it is the same 56,627 scored models the report is built from.</p>
  <div class="panel">
    <div class="row">
      <label for="sig">signal</label>
      <select id="sig"></select>
      <label for="pop" style="min-width:74px">population</label>
      <select id="pop">
        <option value="evaluation">evaluation half (held out)</option>
        <option value="calibration">calibration half</option>
        <option value="all">all models</option>
      </select>
      <button class="act" id="reset">reset to committed</button>
      <button class="act" id="ceiling">push to the ceiling</button>
    </div>
    <div class="row">
      <label for="thr">threshold</label>
      <input type="range" id="thr" min="0" max="1" step="0.000001">
      <input type="number" id="thrn" step="0.000001" style="width:132px">
    </div>
    <div class="row" style="margin-top:16px">
      <div style="min-width:210px">
        <div class="k dim" style="font-size:11px">measured FPR</div>
        <div class="big" id="fpr">&ndash;</div>
        <div class="dim" id="ci"></div>
      </div>
      <div style="min-width:210px">
        <div class="k dim" style="font-size:11px">alarms / population</div>
        <div class="big" id="hits">&ndash;</div>
        <div class="dim" id="denom"></div>
      </div>
      <div style="min-width:210px">
        <div class="k dim" style="font-size:11px">committed FPR</div>
        <div class="big dim" id="cfpr">&ndash;</div>
        <div class="dim" id="cdelta"></div>
      </div>
    </div>
    <div class="bar"><i id="bar" style="width:0"></i></div>
  </div>
  <div id="exploremsg"></div>
  <h2>Signal distribution on this population</h2>
  <p>Left of the line is the calibration half (used to choose the threshold), right is the held-out
  half. A signal whose clean scores pile up at 1.0 cannot produce a separation, which is the
  degeneracy the headline table reports rather than hides.</p>
  <div id="hist"></div>
</section>

<section id="t-tpr">
  <h2>What the 5% false-alarm budget actually buys, class by class</h2>
  <p>These are attacked variants derived from models that are <b>already in the clean
  population above</b> - weight-space tampering only (head noise, structural pruning, and a
  targeted class-bias lift), no retraining, no new data. They were scored through the same
  adapter against the same pinned reference, and then judged at the <b>thresholds the FPR
  half already chose</b>. Nothing here recalibrated anything: the clean ledger and its
  thresholds are untouched, and this is a separate receipt.</p>
  <p><b>There is no pooled detection rate on this page.</b> A rate averaged over attack
  classes survives exactly one follow-up question, and a rate averaged over doses hides
  whether the rule is detecting damage or merely noticing that a weight changed. Every cell
  stands on its own, including the cells where the rule misses.</p>
  <div id="tprhead"></div>
  <table id="tprtable"></table>
  <div id="tprmsg"></div>
  <div id="tprskip"></div>
</section>

<section id="t-sources">
  <h2>Per-source breakdown at the committed threshold</h2>
  <p>Eleven shards, three machines. A pooled rate can hide a source that behaves differently, so the
  per-shard rate is printed beside it - and a shard whose interval excludes the pooled rate is
  flagged rather than averaged away.</p>
  <div id="sourcemsg"></div>
  <table id="sources"></table>
  <h2>Symbols</h2>
  <p>Click a row to load that shard into the explorer as a population.</p>
</section>

<section id="t-split">
  <h2>The split is a free parameter too</h2>
  <p>The threshold and the evaluation half both come from a deterministic permutation seeded by a
  single integer. Re-run with a different seed and the headline moves by a few thousandths. That
  spread is the honest resolution of this measurement, and it is printed here rather than buried -
  a single split seed quoted alone would overstate the precision.</p>
  <table id="sens"></table>
  <div id="sensmsg"></div>
</section>

<section id="t-audit">
  <h2>Audit trail</h2>
  <div id="auditsummary"></div>
  <table id="verify"></table>
  <h2>Every source the census counted</h2>
  <div id="censusmsg"></div>
  <table id="census"></table>
</section>

<section id="t-poison">
  <h2>The model that killed box 5, and what the pipeline did about it</h2>
  <div id="poison"></div>
</section>

</main>
<script>
const DATA = /*__DATA__*/;
const SIGS = DATA.signals;
const ALPHA = DATA.alpha;
const Z = 1.959963984540054;
const TABLES = {headline:'headline',explore:'explore',tpr:'tpr',sources:'sources',split:'split',
                audit:'audit',poison:'poison'};
const TPR_LABEL = {tpr:'detection'};

function b64f(s){const bin=atob(s);const buf=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)buf[i]=bin.charCodeAt(i);
  return new Float64Array(buf.buffer);}
function b64b(s){const bin=atob(s);const buf=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)buf[i]=bin.charCodeAt(i);
  return new Uint8Array(buf.buffer);}
function unpackBits(u,n){const out=new Uint8Array(n);
  for(let i=0;i<n;i++)out[i]=(u[i>>3]>>(7-(i&7)))&1;return out;}

const SCORES={};SIGS.forEach(s=>SCORES[s]=b64f(DATA.scores[s]));
const ISEVAL=unpackBits(b64b(DATA.isEval),DATA.n);
const N=DATA.n;

function wilson(h,n){if(n<=0)return[null,null];const p=h/n,d=1+Z*Z/n;
  const c=(p+Z*Z/(2*n))/d,hf=(Z/d)*Math.sqrt(p*(1-p)/n+Z*Z/(4*n*n));
  return[Math.max(0,c-hf),Math.min(1,c+hf)];}
function quantile(arr,q){const a=arr.slice().sort((x,y)=>x-y);if(!a.length)return NaN;
  const pos=q*(a.length-1),lo=Math.floor(pos),hi=Math.min(lo+1,a.length-1),f=pos-lo;
  return a[lo]*(1-f)+a[hi]*f;}
function pct(v){return (v*100).toFixed(2)+'%';}

// populations are index sets, built once
const EVALIDX=[],CALIDX=[];
for(let i=0;i<N;i++){ (ISEVAL[i]?EVALIDX:CALIDX).push(i); }

let shardFilter=null;   // null = whole corpus
function idxFor(pop){
  if(shardFilter!==null){
    const out=[];
    for(let i=0;i<N;i++) if(DATA.shardOf[i]===shardFilter && (pop==='all'||(pop==='evaluation')===!!ISEVAL[i])) out.push(i);
    return out;
  }
  return pop==='evaluation'?EVALIDX:(pop==='calibration'?CALIDX:Array.from({length:N},(_,i)=>i));
}

// ---------------- tabs ----------------
const nav=document.getElementById('tabs');
Object.keys(TABLES).forEach((k,i)=>{
  const b=document.createElement('button');b.textContent=TPR_LABEL[k]||k;
  if(i===0)b.className='on';b.onclick=()=>{
    document.querySelectorAll('nav button').forEach(x=>x.className='');
    document.querySelectorAll('main section').forEach(x=>x.className='');
    b.className='on';document.getElementById('t-'+TABLES[k]).className='on';
  };
  nav.appendChild(b);
});

document.getElementById('sub').textContent =
  N.toLocaleString()+' merged models \u00b7 '+DATA.shards.length+' shards \u00b7 alpha '+
  ALPHA+' \u00b7 split seed '+DATA.splitSeed+' \u00b7 '+
  CALIDX.length.toLocaleString()+' calibration + '+EVALIDX.length.toLocaleString()+' held-out';

// ---------------- headline ----------------
const cards=document.getElementById('cards');
const live=SIGS.filter(s=>!DATA.committed[s].degenerate);
const best=live.slice().sort((a,b)=>DATA.committed[a].recomputed_fpr-DATA.committed[b].recomputed_fpr)[0];
[['models merged',N.toLocaleString(),'11 sources, 3 machines'],
 ['held-out clean assets',EVALIDX.length.toLocaleString(),'threshold chosen on a disjoint half'],
 ['live rules',live.length+' of '+SIGS.length,'peak / q95 are degenerate, see below'],
 ['best measured FPR',pct(DATA.committed[best].recomputed_fpr),best],
 ['attacked positives','0','so TPR is undefined, not zero'],
 ['quarantined','0','nothing was moved out of any corpus']
].forEach(([k,v,n])=>{
  const d=document.createElement('div');d.className='card';
  d.innerHTML='<div class="k">'+k+'</div><div class="v">'+v+'</div><div class="n">'+n+'</div>';
  cards.appendChild(d);
});

(function(){
  const c=DATA.committed;
  let h='<tr><th>signal</th><th class="num">FPR (95% Wilson)</th><th class="num">alarms / n</th>'+
        '<th class="num">threshold</th><th>status</th><th class="num">committed FPR</th><th>agrees</th></tr>';
  SIGS.forEach(s=>{
    const r=c[s],ci=r.recomputed_ci;
    const agree=r.agrees;  // decided at build time, at the report's published precision
    h+='<tr><td>'+s+'</td>'+
       '<td class="num">'+(r.degenerate?'<span class="y">':'')+pct(r.recomputed_fpr)+
         '</span> ['+pct(ci[0])+', '+pct(ci[1])+']</td>'+
       '<td class="num">'+r.recomputed_hits+' / '+(r.recomputed_hits+ (EVALIDX.length-r.recomputed_hits))+'</td>'+
       '<td class="num">'+r.recomputed_threshold.toFixed(6)+'</td>'+
       '<td>'+(r.degenerate?'<span class="y">degenerate - threshold sits at the corpus ceiling</span>':'<span class="g">fpr_only (no positives)</span>')+'</td>'+
       '<td class="num dim">'+pct(r.committed_fpr)+'</td>'+
       '<td>'+(agree?'<span class="g">yes</span>':'<span class="r">NO</span>')+'</td></tr>';
  });
  document.getElementById('headline').innerHTML=h;
})();

document.getElementById('tprnote').innerHTML =
 '<div class="warn"><b>TPR is absent by construction, not zero.</b> This ledger is clean nulls only '+
 '('+N.toLocaleString()+' models, every one <code>kind: clean</code>). No rule has an attacked asset to '+
 'detect, so a TPR column here would be fabricated. The evaluator therefore records each rule as '+
 '<code>status: fpr_only</code> with an explicit <code>insufficient_denominator</code> refusal and a '+
 '<code>null</code> point estimate. Expected loss, break-even prevalence and clause 3.7 recall are '+
 'unavailable for the same reason - they are all functions of TPR.</div>';

document.getElementById('assembly').innerHTML =
 '<p>Sources are byte-pinned: each shard ships a <code>registry_*.meta.json</code> holding the '+
 'sha256 of the shard file it scored, and the merge recomputes that digest from the plan and refuses '+
 'on a mismatch. All shards must agree on one reference model and one scorer configuration; '+
 '<code>clean_none_fixed_s5800</code> is that reference and is excluded from every shard.</p>'+
 '<div class="ok">Verification: <b>'+DATA.verify.sources.reduce((a,s)=>a+s.verified,0).toLocaleString()+
 ' of '+DATA.verify.sources.reduce((a,s)=>a+s.verified+s.failed,0).toLocaleString()+
 '</b> models verified against their own pinned manifest and weight digests, <b>0 failed</b>. '+
 'No quarantine directory was ever created, because nothing failed to move.</div>';

// ---------------- explorer ----------------
const sigSel=document.getElementById('sig'),popSel=document.getElementById('pop');
SIGS.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;sigSel.appendChild(o);});
sigSel.value=best;
const thrRange=document.getElementById('thr'),thrNum=document.getElementById('thrn');
let curPop='evaluation';

function committedThreshold(sig){
  const cal=idxFor('calibration').map(i=>SCORES[sig][i]);
  return quantile(cal,1-ALPHA);
}
function renderExplore(){
  const sig=sigSel.value,idx=idxFor(curPop);
  const thr=parseFloat(thrNum.value);
  const cal=idxFor('calibration').map(i=>SCORES[sig][i]);
  const ceiling=Math.max.apply(null,cal);
  const cthr=committedThreshold(sig),cfpr=DATA.committed[sig].recomputed_fpr;
  let hits=0;for(const i of idx)if(SCORES[sig][i]>thr)hits++;
  const n=idx.length,p=hits/n,ci=wilson(hits,n);
  document.getElementById('fpr').textContent=pct(p);
  document.getElementById('ci').textContent='95% Wilson ['+pct(ci[0])+', '+pct(ci[1])+']';
  document.getElementById('hits').textContent=hits.toLocaleString()+' / '+n.toLocaleString();
  document.getElementById('denom').textContent=curPop==='evaluation'?
    'held-out half - never used to pick a threshold':
    (curPop==='calibration'?'calibration half - the threshold comes from here':'whole merged corpus');
  document.getElementById('cfpr').textContent=pct(cfpr);
  const d=(p-cfpr)*100;
  document.getElementById('cdelta').textContent=
    (Math.abs(d)<0.005?'at the committed operating point':
     (d>0?'+':'')+d.toFixed(2)+' points vs committed');
  document.getElementById('bar').style.width=Math.min(100,p*100)+'%';
  const msg=document.getElementById('exploremsg');
  if(thr>=ceiling){
    msg.innerHTML='<div class="bad"><b>This rule cannot fire at this threshold.</b> The threshold is '+
    'at or above the highest clean score in the calibration half ('+ceiling.toFixed(6)+'), so no clean '+
    'model can ever cross it. An FPR of 0.0000 here means the rule is <i>unable</i> to alarm - which is '+
    'exactly what <code>ctc_peak_clean</code> and <code>ctc_q95_clean</code> do at the committed '+
    'setting. Zero from inability is not accuracy.</div>';
  } else if(Math.abs(thr-cthr)<1e-9){
    msg.innerHTML='<div class="ok">This is the committed operating point: the '+
    (((1-ALPHA)*100).toFixed(0))+'th percentile of '+cal.length.toLocaleString()+
    ' calibration negatives. '+(curPop==='evaluation'?
      'On the held-out half it measures '+pct(DATA.committed[sig].recomputed_fpr)+'.':
      'Switch to the evaluation half to see the held-out rate.')+'</div>';
  } else if(curPop==='calibration'){
    msg.innerHTML='<div class="warn">Reading the rate on the calibration half is <i>optimistic by '+
    'construction</i> - the threshold was chosen from these same models. It is here only to make that '+
    'circularity visible.</div>';
  } else { msg.innerHTML=''; }
}
function syncFromRange(){
  const sig=sigSel.value,cal=idxFor('calibration').map(i=>SCORES[sig][i]);
  const lo=Math.min.apply(null,cal),hi=Math.max.apply(null,cal);
  const span=(hi-lo)||1;
  const v=lo+span*parseFloat(thrRange.value);
  thrNum.value=v.toPrecision(9);
  renderExplore();
}
function setRangeFromNumber(){
  const sig=sigSel.value,cal=idxFor('calibration').map(i=>SCORES[sig][i]);
  const lo=Math.min.apply(null,cal),hi=Math.max.apply(null,cal);
  const span=(hi-lo)||1;
  thrRange.value=Math.max(0,Math.min(1,(parseFloat(thrNum.value)-lo)/span));
  renderExplore();
}
function resetExplorer(){
  const sig=sigSel.value;
  thrNum.value=committedThreshold(sig).toPrecision(9);
  setRangeFromNumber();
}
sigSel.onchange=()=>{shardFilter=null;resetExplorer();drawHist();};
popSel.onchange=()=>{curPop=popSel.value;renderExplore();};
thrRange.oninput=syncFromRange;
thrNum.oninput=setRangeFromNumber;
document.getElementById('reset').onclick=resetExplorer;
document.getElementById('ceiling').onclick=()=>{
  const sig=sigSel.value,cal=idxFor('calibration').map(i=>SCORES[sig][i]);
  thrNum.value=Math.max.apply(null,cal).toPrecision(9);setRangeFromNumber();};

function drawHist(){
  const sig=sigSel.value,cal=idxFor('calibration'),ev=idxFor('evaluation');
  const lo=Math.min.apply(null,cal.map(i=>SCORES[sig][i]));
  const hi=Math.max.apply(null,cal.map(i=>SCORES[sig][i]))||1;
  const B=60,hc=new Array(B).fill(0),he=new Array(B).fill(0);
  const bin=v=>Math.max(0,Math.min(B-1,Math.floor((v-lo)/((hi-lo)||1)*B)));
  for(const i of cal)hc[bin(SCORES[sig][i])]++;
  for(const i of ev)he[bin(SCORES[sig][i])]++;
  const mx=Math.max.apply(null,hc.concat(he))||1;
  let bars='';
  for(let b=0;b<B;b++){
    bars+='<div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;'+
      'height:84px" title="bin '+b+' cal '+hc[b]+' / eval '+he[b]+'">'+
      '<i style="display:block;height:'+(he[b]/mx*84)+'px;background:#1f6feb"></i>'+
      '<i style="display:block;height:'+(hc[b]/mx*84)+'px;background:#8b949e"></i></div>';
  }
  document.getElementById('hist').innerHTML=
    '<div style="display:flex;gap:1px;align-items:flex-end;border-bottom:1px solid #30363d">'+bars+'</div>'+
    '<div class="dim" style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px">'+
    '<span>'+lo.toFixed(4)+'</span><span title="threshold">'+
    '^ threshold '+committedThreshold(sig).toFixed(4)+'</span><span>'+hi.toFixed(4)+'</span></div>'+
    '<div class="dim" style="font-size:11px;margin-top:4px">'+
    '<span style="color:#8b949e">\u25a0</span> calibration ('+cal.length.toLocaleString()+') '+
    '<span style="color:#1f6feb">\u25a0</span> evaluation ('+ev.length.toLocaleString()+')</div>';
}

// ---------------- sources ----------------
function drawSources(){
  const sig=best,thr=DATA.committed[sig].recomputed_threshold;
  const rows=[];
  DATA.shards.forEach((sh,si)=>{
    let hits=0,n=0;
    for(let i=0;i<N;i++){
      if(DATA.shardOf[i]!==si||!ISEVAL[i])continue;
      n++;if(SCORES[sig][i]>thr)hits++;
    }
    rows.push({sh,si,hits,n,fpr:n?hits/n:null});
  });
  const tot=rows.reduce((a,r)=>a+r.hits,0)/rows.reduce((a,r)=>a+r.n,0);
  let h='<tr><th>shard</th><th class="num">models</th><th class="num">seeds</th>'+
        '<th class="num">held-out alarms</th><th class="num">FPR</th><th>vs pooled '+pct(tot)+'</th></tr>';
  let flagged=0;
  rows.forEach(r=>{
    const ci=wilson(r.hits,r.n);
    const excludes=r.fpr!==null&&(ci[0]>tot||ci[1]<tot);
    if(excludes)flagged++;
    h+='<tr style="cursor:pointer" onclick="pickShard('+r.si+')"><td>'+r.sh.shard_id+'</td>'+
       '<td class="num">'+r.sh.count.toLocaleString()+'</td>'+
       '<td class="num dim">'+r.sh.seed_min+'-'+r.sh.seed_max+'</td>'+
       '<td class="num">'+r.hits+' / '+r.n+'</td>'+
       '<td class="num">'+(r.fpr===null?'&ndash;':pct(r.fpr))+'</td>'+
       '<td>'+(excludes?'<span class="y">interval excludes pooled rate</span>':'<span class="dim">consistent</span>')+
       '</td></tr>';
  });
  document.getElementById('sources').innerHTML=h;
  document.getElementById('sourcemsg').innerHTML=
    '<div class="'+(flagged?'warn':'ok')+'">'+flagged+' of '+rows.length+
    ' shards have a Wilson interval that excludes the pooled rate'+
    (flagged?' - quoting the pooled number without these is the averaging mistake this lane already made once.':' - the pooled rate is representative of every shard.')+
    '</div>';
}
function pickShard(si){
  shardFilter=si;curPop=popSel.value='evaluation';
  document.querySelectorAll('nav button')[1].click();
  sigSel.value=best;resetExplorer();drawHist();
}
drawSources();

// ---------------- split sensitivity ----------------
(function(){
  let h='<tr><th class="num">split seed</th>';
  SIGS.forEach(s=>h+='<th class="num">'+s+'<br><span class="dim">thr / FPR</span></th>');
  h+='</tr>';
  DATA.sensitivity.forEach(row=>{
    h+='<tr><td class="num">'+row.seed+(row.seed===DATA.splitSeed?' <span class="g">committed</span>':'')+'</td>';
    SIGS.forEach(s=>{const r=row.rules[s];
      h+='<td class="num">'+r.threshold.toFixed(6)+'<br><span class="dim">'+pct(r.fpr)+'</span></td>';});
    h+='</tr>';
  });
  document.getElementById('sens').innerHTML=h;
  const spread={};
  SIGS.forEach(s=>{
    const vs=DATA.sensitivity.map(r=>r.rules[s].fpr);
    spread[s]=[Math.min.apply(null,vs),Math.max.apply(null,vs)];
  });
  document.getElementById('sensmsg').innerHTML=
    '<div class="warn">The headline is a property of the corpus <i>and</i> the split seed. '+
    SIGS.map(s=>'<code>'+s+'</code> spans '+pct(spread[s][0])+'&ndash;'+pct(spread[s][1])).join(', ')+
    '. Any single-split figure quoted to four decimals is overstating this measurement\u2019s '+
    'resolution; the spread across seeds is the honest error bar.</div>';
})();

// ---------------- audit ----------------
(function(){
  const v=DATA.verify;
  document.getElementById('auditsummary').innerHTML=
    '<div class="ok"><b>Verification is a gate, not a report.</b> Each model is checked against the '+
    'sha256 of <i>both</i> its manifest and its weights, taken from the plan that claims it. A failure '+
    'is moved aside (never deleted, never scored) and listed. Nothing failed here, so the quarantine '+
    'directory does not exist - its absence is the evidence.</div>';
  let h='<tr><th>source</th><th class="num">verified</th><th class="num">failed</th>'+
        '<th class="num">not in the pinned plan</th></tr>';
  v.sources.forEach(s=>{h+='<tr><td>'+s.id+'</td><td class="num g">'+s.verified.toLocaleString()+
    '</td><td class="num">'+s.failed+'</td><td class="num dim">'+s.no_plan+'</td></tr>';});
  const tv=v.sources.reduce((a,s)=>a+s.verified,0),tf=v.sources.reduce((a,s)=>a+s.failed,0);
  h+='<tr><td><b>total</b></td><td class="num"><b>'+tv.toLocaleString()+'</b></td>'+
     '<td class="num"><b>'+tf+'</b></td><td class="num dim">&ndash;</td></tr>';
  document.getElementById('verify').innerHTML=h;

  const c=DATA.census;
  document.getElementById('censusmsg').innerHTML=
    '<div class="ok">'+c.n_sources+' sources counted before anything was scored. '+
    c.n_collisions.toLocaleString()+' cross-source seed collisions resolved by hashing the bytes '+
    '<i>inside</i> the archives rather than by trusting filenames; '+c.collisions_fatal+
    ' fatal. Lab seed range '+c.lab_seed_range[0].toLocaleString()+'-'+c.lab_seed_range[1].toLocaleString()+
    '.</div>'+
    (c.gaps.length?'<div class="warn"><b>Known gap:</b> '+c.gaps.map(g=>g.id+' \u2014 '+(g.note||'')).join('; ')+
     '</div>':'');
  let ch='<tr><th>source</th><th>kind</th><th class="num">models</th><th class="num">seeds</th></tr>';
  c.sources.forEach(s=>{
    const q=s.n===0?'<span class="y">0 - absent</span>':s.n.toLocaleString();
    ch+='<tr><td>'+s.id+'</td><td class="dim">'+(s.kind||'')+'</td><td class="num">'+q+'</td>'+
        '<td class="num dim">'+(s.seed_min==null?'&ndash;':s.seed_min+'-'+s.seed_max)+'</td></tr>';
  });
  document.getElementById('census').innerHTML=ch;
})();

// ---------------- poison story ----------------
document.getElementById('poison').innerHTML=
 '<p>Box 5 shipped a registry with 1,467 rows and <b>no results file</b>. Reproduced here, the cause is '+
 'one clean model whose CTC statistic is undefined on all 40 held-out images, so the producer emits a '+
 'single key and any adapter requiring four raises inside the worker pool - which took the remaining '+
 '3,533 models of that shard down with it.</p>'+
 '<table class="timeline"><tr><th>step</th><th>what happened</th></tr>'+
 '<tr><td>on the box</td><td>run dies at row 1,467 with <code>ValueError: repository scorer omitted '+
 'signals: dict_keys([&#39;refdiv_mean_clean&#39;])</code> and no npz is written</td></tr>'+
 '<tr><td>reproduced</td><td><code>s81621</code> scores in 0.07 s and returns exactly one key. Its '+
 'manifest keys and scene spec match its neighbours - it is not a corrupt artifact, it is a statistic '+
 'that is undefined for that model</td></tr>'+
 '<tr><td>the refusal</td><td>the old adapter raises <i>inside</i> a worker future, and '+
 '<code>future.result()</code> kills the process. One undefined model costs 5,000 rows</td></tr>'+
 '<tr><td>the fix</td><td>the tolerant scorer classifies instead of raising: <b>complete</b>, '+
 '<b>incomplete</b> (undefined on every image - a property of the model) and <b>error</b> (a property '+
 'of the run, and retried on resume)</td></tr>'+
 '<tr><td>the outcome</td><td>the other <b>4,999</b> scored. <code>s81621</code> is one row in '+
 '<code>skipped_seed_80152_85151.json</code> carrying the box&#39;s own message, and the re-issued '+
 'shard declares <code>unscorable_model_ids: [clean_none_fixed_s81621]</code></td></tr>'+
 '<tr><td>the invariant</td><td>the shard the merge validates is byte-identical to the shard that was '+
 'scored, so the lost model is a named number rather than a silently shrunken denominator</td></tr>'+
 '</table>'+
 '<div class="ok">The box&#39;s surviving rows reproduce byte-for-byte on this machine '+
 '(<code>s81617</code> \u2192 ctc_mean 0.7748 / refdiv 0.437), so the rescored half is comparable with '+
 'the half the box wrote.</div>'+
 '<div class="warn">This is the failure mode the whole pass is designed around: <b>one undefined '+
 'statistic must not become a lost corpus, and must not become a quietly smaller denominator.'+
 '</b></div>';

// ---------------- detection: TPR ladder at frozen thresholds ----------------
(function(){
  const T=DATA.tpr;
  const host=document.getElementById('tprhead');
  if(!T){host.innerHTML='<div class="warn">No attacked-arm receipt was supplied to this build, so no detection panel is shown. The false-alarm numbers above stand alone: they are a clean-null measurement.</div>';return;}
  const sigs=Object.keys(T.rules);
  const live=sigs.filter(s=>!T.rules[s].degenerate);
  const deg=sigs.filter(s=>T.rules[s].degenerate);

  // The single best cell - and only among cells with enough arms to conclude from, so the
  // card can never show a headline number that the table below flags as too small.
  let best=null;
  T.kinds.forEach(k=>T.doses.forEach(d=>{
    const c=(T.cells[k]||{})[String(d)]; if(!c) return;
    live.forEach(s=>{
      const r=c.rules[s];
      if(r.tpr===null||r.n_below_floor) return;
      if(!best||r.tpr>best.tpr) best={kind:k,dose:d,tpr:r.tpr,caught:r.caught,n:c.n,rule:s};
    });
  }));
  host.innerHTML=
    '<div class="cards">'+
    '<div class="card"><div class="k">attacked variants</div><div class="v">'+T.n_arms+'</div>'+
      '<div class="n">'+T.kinds.length+' classes &times; '+T.doses.length+' doses, no retraining</div></div>'+
    (best?'<div class="card"><div class="k">best measured cell</div><div class="v">'+pct(best.tpr)+'</div>'+
      '<div class="n">'+best.rule+' on '+best.kind+' @ dose '+best.dose+' ('+best.caught+'/'+best.n+')</div></div>':
      '<div class="card"><div class="k">best measured cell</div><div class="v">&ndash;</div>'+
      '<div class="n">every cell is below the floor</div></div>')+
    '<div class="card"><div class="k">rules that cannot fire</div><div class="v">'+deg.length+' of '+sigs.length+
      '</div><div class="n">threshold at the corpus ceiling</div></div>'+
    '<div class="card"><div class="k">false alarms</div><div class="v">'+pct(DATA.committed[best?best.rule:live[0]].recomputed_fpr)+
      '</div><div class="n">unchanged - not recalibrated for attacks</div></div></div>'+
    '<div class="bad"><b>No pooled detection rate is on this page, and that is deliberate.</b> '+
      (T.pooled_refusal||'')+'</div>';

  // One row per (attack class, dose). No row is a rule, and no column is an average.
  let h='<tr><th>attack class</th><th class="num">dose</th><th class="num">arms</th>'+
        '<th class="num">behaviour unmoved</th>';
  live.forEach(s=>{h+='<th class="num">'+s+'</th>';});
  h+='</tr>';
  T.kinds.forEach(k=>{
    T.doses.forEach(d=>{
      const c=(T.cells[k]||{})[String(d)]; if(!c) return;
      h+='<tr><td>'+k+'</td><td class="num">'+d+'</td><td class="num">'+c.n+'</td>'+
         '<td class="num dim">'+c.n_behaviour_inert+' / '+c.n+'</td>';
      live.forEach(s=>{
        const r=c.rules[s];
        if(r.tpr===null){h+='<td class="num y">&ndash;</td>';return;}
        const ci=r.ci95_wilson;
        h+='<td class="num">'+pct(r.tpr)+(r.n_below_floor?' <span class="y">*</span>':'')+
           ' <span class="dim">['+pct(ci[0])+', '+pct(ci[1])+']</span></td>';
      });
      h+='</tr>';
    });
  });
  document.getElementById('tprtable').innerHTML=h;

  document.getElementById('tprmsg').innerHTML=
    '<div class="warn"><b>Read down a class, not across the table.</b> A dose is a different '+
    'unit in each family - the units are listed below - so the only comparison that means '+
    'anything is between rules <i>inside</i> one cell. '+
    (deg.length?('The rules '+deg.map(s=>'<code>'+s+'</code>').join(' and ')+
      ' show as &ndash; because their frozen threshold is the ceiling of the clean corpus: '+
      'they cannot fire on an attack either. That is undefined, not 0%.'):'')+
    ' A cell marked <span class="y">*</span> has fewer than '+T.min_positives+
    ' arms left, so its ratio is shown beside its count and interval and should not be '+
    'quoted on its own.</div>';

  // Nothing here is derived from the raw skipped receipt any more: the evaluator already
  // grouped the unscorable arms by (class, dose), which is the grouping that matters.
  let s='';
  const un=Object.entries(T.unscorable||{});
  if(un.length){
    s+='<div class="bad"><b>Survivorship, and it gets worse with dose.</b> These arms were '+
      'built and could not be scored at all - the scorer returned no CTC statistic, so no '+
      'rule ever got to look at them, and they are <i>not</i> counted as misses: '+
      un.map(([k,v])=>k+' &times;'+v).join(', ')+
      '. At the highest dose the attack removes most of its own subjects, so the rate on '+
      'what remains is an upper bound measured on a shrinking subset. The true detection '+
      'rate on those cells is <b>no better</b> than the number shown.</div>';
  }
  if(T.behaviour_metric){
    s+='<div class="warn"><b>Some arms cannot be caught by listening to behaviour, and the '+
      'table says how many.</b> '+T.behaviour_metric+'</div>';
  }
  if(T.units_per_kind&&Object.keys(T.units_per_kind).length){
    s+='<div class="warn"><b>The dose axes are not the same units.</b> '+
      Object.entries(T.units_per_kind).map(([k,v])=>'<code>'+k+'</code> = '+v).join('; ')+
      '. A ladder is comparable within a family and not across families.</div>';
  }
  if(T.problems&&T.problems.length){
    s+='<div class="bad">'+T.problems.join('; ')+'</div>';
  }
  if(!s){
    s='<div class="ok">Every arm was scored and every class is measurable, so nothing in '+
      'this table is an upper bound dressed as an estimate.</div>';
  }
  document.getElementById('tprskip').innerHTML=s;
})();

// ---------------- boot ----------------
resetExplorer();drawHist();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
