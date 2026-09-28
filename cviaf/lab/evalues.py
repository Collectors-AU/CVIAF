"""Conservative, fixed-calibration e-processes for repeated asset monitoring.

Under iid (or conditionally bounded-tail) null scores, DKW gives a simultaneous
upper bound on the true exceedance probability with probability >= 1-delta.
Conditional on this calibration event, each fixed-stake binary bet is an e-value.
Thus the product of fused e-values is a nonnegative supermartingale; its running
maximum crosses 1/alpha with probability at most alpha, when threshold is 1/(alpha-delta). Calibration
scores must be separate from monitored scores and frozen for the whole ledger.
Reusing a small finite calibration sample to create raw conformal p-values at
successive times does NOT establish conditional e-value validity.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class TailBet:
    cutoff: float
    tail_bound: float
    stake: float
    calibration_digest: str
    n_cal: int

    @classmethod
    def calibrate(cls, scores, *, delta: float, quantile: float = .75,
                  stake: float = 0.5) -> 'TailBet':
        x = np.asarray(scores, dtype=float).ravel()
        if x.size < 2 or not np.all(np.isfinite(x)):
            raise ValueError('calibration scores must contain at least two finite values')
        if not (0 < delta < 1 and 0 < quantile < 1 and stake > 0):
            raise ValueError('invalid calibration policy')
        cutoff = float(np.quantile(x, quantile, method='higher'))
        # One-sided DKW is <= sqrt(log(1/delta)/(2n)); use two-sided
        # constant for a conservative bound and a simultaneous CDF event.
        eps = math.sqrt(math.log(2 / delta) / (2 * x.size))
        bound = min(1.0, float(np.mean(x > cutoff)) + eps)
        if stake * bound > 1:
            raise ValueError('stake would make e-value negative')
        digest = hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()
        return cls(cutoff, bound, stake, digest, x.size)

    def value(self, score: float) -> float:
        if not math.isfinite(score):
            raise ValueError('missing or nonfinite monitored score')
        return 1.0 + self.stake * (float(score > self.cutoff) - self.tail_bound)


@dataclass
class WealthLedger:
    """One ledger per asset/hypothesis; not a posterior or an FDR claim."""
    asset_id: str
    bets: Mapping[str, TailBet]
    alpha: float = .05
    delta: float = .005
    weights: Mapping[str, float] | None = None
    log_wealth: float = 0.0
    high_water_log: float = 0.0
    first_alarm: int | None = None
    records: list[dict] = field(default_factory=list)
    _previous_hash: str = '0' * 64

    def __post_init__(self):
        if not self.asset_id or not self.bets or not (0 < self.delta < self.alpha < 1):
            raise ValueError('asset ID, bets and alpha are required')
        w = self.weights or {k: 1. for k in self.bets}
        if set(w) != set(self.bets) or any(not math.isfinite(v) or v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError('weights must be fixed, finite and nonnegative')
        self.weights = {k: v / sum(w.values()) for k, v in w.items()}

    def step(self, scores: Mapping[str, float]) -> dict:
        if set(scores) != set(self.bets):
            raise ValueError('all predeclared detectors must report each time; do not drop failures')
        if any(not math.isfinite(b.tail_bound) or not (0 <= b.tail_bound <= 1) or b.stake * b.tail_bound >= 1 for b in self.bets.values()):
            raise ValueError('bets must have strictly positive e-values')
        e = {k: bet.value(float(scores[k])) for k, bet in self.bets.items()}
        merged = sum(self.weights[k] * e[k] for k in self.bets)
        if merged < 0:
            raise ValueError('negative e-value')
        self.log_wealth += math.log(merged) if merged else -math.inf
        self.high_water_log = max(self.high_water_log, self.log_wealth)
        t = len(self.records) + 1
        if self.first_alarm is None and self.high_water_log >= math.log(1 / (self.alpha - self.delta)):
            self.first_alarm = t
        record = {'asset_id': self.asset_id, 't': t, 'scores': dict(scores),
                  'e_values': e, 'fused_e': merged,
                  'log_wealth': self.log_wealth,
                  'wealth': math.exp(self.log_wealth),
                  'threshold': 1 / (self.alpha - self.delta),
                  'alarm': self.first_alarm is not None,
                  'time_to_alarm': self.first_alarm,
                  'calibration': {k: b.calibration_digest for k, b in self.bets.items()},
                  'previous_hash': self._previous_hash}
        self._previous_hash = hashlib.sha256(json.dumps(record, sort_keys=True,
                                             allow_nan=False).encode()).hexdigest()
        record['record_hash'] = self._previous_hash
        self.records.append(record)
        return record
