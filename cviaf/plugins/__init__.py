"""Detector protocol and deterministic registry. No engine or optional ML imports."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Literal
import math

REASONS = frozenset({'ACCESS_DENIED', 'ASSET_ABSENT', 'REFERENCE_MISSING',
                     'BUDGET_EXHAUSTED', 'NOT_APPLICABLE', 'DEPENDENCY_MISSING',
                     'NO_GPU_OR_UNMET_VRAM', 'EXECUTION_ERROR'})

@dataclass(frozen=True)
class CostModel:
    forward_passes_per_unit: int = 0
    wall_clock_class: str = 'short'
    gpu_required: bool = False
    seconds: float = 0.0
    vram_gb: float = 0.0

    def __post_init__(self):
        if self.forward_passes_per_unit < 0 or not self.wall_clock_class or any(
            not math.isfinite(x) or x < 0 for x in (self.seconds, self.vram_gb)
        ):
            raise ValueError('invalid cost model')

@dataclass(frozen=True)
class CalibrationRef:
    slice_id: str
    digest: str
    def __post_init__(self):
        if not self.slice_id or not self.digest:
            raise ValueError('calibration requires a battery slice and digest')

@dataclass(frozen=True)
class Manifest:
    id: str
    module: str
    attack_classes: tuple[str, ...]
    access_required: frozenset[str]
    cost: CostModel
    calibration: CalibrationRef
    emits: Literal['p_value', 'e_value', 'statistic_with_floor']
    formats: frozenset[str] = frozenset()  # empty means format-independent
    battery_slices: frozenset[str] = frozenset()
    residual_risk: str = 'Unmeasured residual risk'

    def __post_init__(self):
        if not self.id or not self.module or not self.attack_classes or not all(self.attack_classes):
            raise ValueError('manifest needs identity and attack classes')
        if len(set(self.attack_classes)) != len(self.attack_classes):
            raise ValueError('duplicate attack class')
        if self.emits not in ('p_value', 'e_value', 'statistic_with_floor'):
            raise ValueError('invalid emission type')
        if not self.access_required <= {'dataset', 'model.black_box', 'model.white_box',
                                        'log', 'ref_dist', 'ops', 'ops.labels'}:
            raise ValueError('unknown access requirement')

@dataclass(frozen=True)
class Result:
    detector: str
    status: Literal['ran', 'skipped', 'error']
    statistic: float | None = None
    p_value: float | None = None
    e_value: float | None = None
    detection_floor: Mapping[str, Any] | None = None
    calibration_ref: str | None = None
    cost_actual: float | None = None
    evidence: tuple[str, ...] = ()
    reason_code: str | None = None
    human_reason: str | None = None
    coverage_impact: str | None = None

    def __post_init__(self):
        if not self.detector or self.status not in ('ran', 'skipped', 'error'):
            raise ValueError('invalid detector/status')
        if self.status == 'ran':
            if self.reason_code is not None or self.statistic is None or not math.isfinite(self.statistic):
                raise ValueError('ran requires finite statistic and no skip reason')
            if (self.p_value is not None) + (self.e_value is not None) + (self.detection_floor is not None) != 1:
                raise ValueError('ran requires exactly one calibrated emission')
            if self.p_value is not None and (not math.isfinite(self.p_value) or not 0 <= self.p_value <= 1):
                raise ValueError('invalid p-value')
            if self.e_value is not None and (not math.isfinite(self.e_value) or self.e_value < 0):
                raise ValueError('invalid e-value')
            if not self.calibration_ref or self.cost_actual is None or not math.isfinite(self.cost_actual) or self.cost_actual < 0 or not self.evidence:
                raise ValueError('ran requires calibration, cost and evidence')
        elif self.reason_code not in REASONS or not self.human_reason or not self.coverage_impact:
            raise ValueError('skip/error requires coded reason, explanation and impact')

class DetectorPlugin(Protocol):
    manifest: Manifest
    def run(self, assets: Mapping[str, Any], battery_slice: Mapping[str, Any],
            budget: Mapping[str, Any]) -> Result: ...

class Registry:
    def __init__(self):
        self._plugins: dict[str, DetectorPlugin] = {}

    def register(self, plugin: DetectorPlugin) -> None:
        manifest = plugin.manifest
        if manifest.id in self._plugins or not callable(getattr(plugin, 'run', None)):
            raise ValueError(f'duplicate or invalid plugin: {manifest.id}')
        self._plugins[manifest.id] = plugin

    def manifests(self) -> tuple[Manifest, ...]:
        return tuple(self._plugins[key].manifest for key in sorted(self._plugins))

    def execute(self, selected: tuple[str, ...] | list[str], assets: Mapping[str, Any],
                battery_slices: Mapping[str, Mapping[str, Any]], budget: Mapping[str, Any]) -> tuple[Result, ...]:
        out = []
        for key in selected:
            plugin = self._plugins[key]
            manifest = plugin.manifest
            try:
                result = plugin.run(assets, battery_slices[manifest.calibration.slice_id], budget)
                if not isinstance(result, Result) or result.detector != key:
                    raise ValueError('plugin returned invalid detector result')
                if result.status == 'ran' and (result.calibration_ref != manifest.calibration.digest or
                    (manifest.emits == 'p_value' and result.p_value is None) or
                    (manifest.emits == 'e_value' and result.e_value is None) or
                    (manifest.emits == 'statistic_with_floor' and result.detection_floor is None)):
                    raise ValueError('emission or calibration digest does not match manifest')
            except Exception as exc:
                result = Result(key, 'error', reason_code='EXECUTION_ERROR',
                                human_reason=f'{type(exc).__name__}: {exc}',
                                coverage_impact='check not executed; no coverage credit')
            out.append(result)
        return tuple(out)
