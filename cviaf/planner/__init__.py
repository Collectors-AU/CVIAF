"""Access-aware AB-1 planner. Outputs decisions, never silently drops checks."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping
from cviaf.plugins import Manifest, Result

@dataclass(frozen=True)
class Context:
    attack_classes: frozenset[str]
    access: frozenset[str]
    formats: frozenset[str]
    assets: frozenset[str]
    battery_slices: frozenset[str]
    seconds: float
    gpu: bool = False
    vram_gb: float = 0.0
    risk_weights: Mapping[str, float] | None = None

    def __post_init__(self):
        if self.seconds < 0 or self.vram_gb < 0 or not self.attack_classes:
            raise ValueError('invalid planning context')
        if self.risk_weights and any(v < 0 for v in self.risk_weights.values()):
            raise ValueError('risk weights must be nonnegative')

@dataclass(frozen=True)
class Rejection:
    detector: str
    reason: str
    coverage_impact: str

@dataclass(frozen=True)
class Plan:
    planner_version: str
    selected: tuple[str, ...]
    rejected: tuple[Rejection, ...]
    attack_classes_in_scope: tuple[str, ...]
    uncovered: tuple[str, ...]
    budget_seconds: float

    def to_dict(self) -> dict:
        return {'planner_version': self.planner_version, 'selected': list(self.selected),
                'rejected': [vars(x) for x in self.rejected], 'uncovered': list(self.uncovered),
                'attack_classes_in_scope': list(self.attack_classes_in_scope),
                'budget_seconds': self.budget_seconds}

def _ineligible(m: Manifest, c: Context) -> str | None:
    if not c.attack_classes.intersection(m.attack_classes):
        return 'NOT_APPLICABLE'
    if not m.access_required <= c.access:
        return 'ACCESS_DENIED'
    if not m.access_required <= c.assets:
        return 'REFERENCE_MISSING' if 'ref_dist' in m.access_required - c.assets else 'ASSET_ABSENT'
    if m.formats and not m.formats.intersection(c.formats):
        return 'ASSET_ABSENT'
    if not (m.battery_slices | {m.calibration.slice_id}) <= c.battery_slices:
        return 'REFERENCE_MISSING'
    if m.cost.gpu_required and (not c.gpu or c.vram_gb < m.cost.vram_gb):
        return 'NO_GPU_OR_UNMET_VRAM'
    return None

def plan(manifests: Iterable[Manifest], context: Context) -> Plan:
    """Maximize weighted covered classes, then breadth, then minimize cost.

    Exact search for <=20 eligible checks; deterministic greedy marginal gain thereafter.
    Every alternative is logged, including overlap and budget rejects.
    """
    ms = sorted(manifests, key=lambda m: m.id)
    if len({m.id for m in ms}) != len(ms):
        raise ValueError('duplicate detector id')
    rejected: list[Rejection] = []
    eligible = []
    for m in ms:
        reason = _ineligible(m, context)
        if reason:
            rejected.append(Rejection(m.id, reason, f'{sorted(context.attack_classes.intersection(m.attack_classes))}: no coverage credit'))
        else:
            eligible.append(m)
    weights = context.risk_weights or {}
    def score(subset):
        classes = set().union(*(set(m.attack_classes) for m in subset)) & context.attack_classes if subset else set()
        return (sum(weights.get(k, 1.0) for k in classes), len(classes),
                -sum(m.cost.seconds for m in subset), tuple(-ord(x) for x in ''.join(m.id for m in subset)))
    if len(eligible) <= 20:
        selected = max((s for n in range(len(eligible)+1) for s in combinations(eligible, n)
                        if sum(m.cost.seconds for m in s) <= context.seconds), key=score)
    else:
        chosen = []
        remaining = eligible[:]
        while remaining:
            affordable = [m for m in remaining if sum(x.cost.seconds for x in chosen) + m.cost.seconds <= context.seconds]
            if not affordable:
                break
            best = max(affordable, key=lambda m: (score(chosen+[m])[0]-score(chosen)[0],
                                                     -m.cost.seconds, m.id))
            if score(chosen+[best])[0] <= score(chosen)[0]:
                break
            chosen.append(best)
            remaining.remove(best)
        selected = tuple(chosen)
    selected_ids = {m.id for m in selected}
    covered = set().union(*(set(m.attack_classes) for m in selected)) & context.attack_classes if selected else set()
    for m in eligible:
        if m.id not in selected_ids:
            reason = 'BUDGET_EXHAUSTED' if sum(x.cost.seconds for x in selected)+m.cost.seconds > context.seconds else 'NOT_SELECTED_OVERLAP'
            rejected.append(Rejection(m.id, reason, f'{sorted(set(m.attack_classes) & context.attack_classes)}: not executed'))
    return Plan('1.0', tuple(m.id for m in selected), tuple(sorted(rejected, key=lambda x: x.detector)),
                tuple(sorted(context.attack_classes)), tuple(sorted(context.attack_classes - covered)), context.seconds)

def completeness(plan: Plan, results: Iterable[Result]) -> dict:
    results = tuple(results)
    by_id = {r.detector: r for r in results}
    if len(by_id) != len(results) or set(by_id) != set(plan.selected):
        raise ValueError('results must match plan exactly')
    applicable = len(plan.selected) + sum(r.reason != 'NOT_APPLICABLE' for r in plan.rejected)
    executed = sum(r.status == 'ran' for r in by_id.values())
    blocking = [f'{r.detector}: {r.reason_code}' for r in by_id.values() if r.status != 'ran'] + [f'{r.detector}: {r.reason}' for r in plan.rejected if r.reason != 'NOT_APPLICABLE']
    return {'applicable': applicable, 'executed': executed,
            'fraction': executed / applicable if applicable else 0.0,
            'accept_permitted': bool(applicable and not blocking and not plan.uncovered),
            'blocking_skips': sorted(blocking)}
