"""Fixed v6 feature schema and explicit metadata audit."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str = "float64"
    unit: str = "dimensionless"
    block: str = "unknown"
    definition: str = ""
    structural_zero_allowed: bool = True
    missing_allowed: bool = False


@dataclass
class FeatureSchema:
    scenario: str
    version: str
    features: Dict[str, FeatureSpec]

    def validate(self, values: Mapping[str, object]) -> Dict[str, object]:
        missing = [name for name, spec in self.features.items() if name not in values and not spec.missing_allowed]
        unknown = sorted(set(values) - set(self.features))
        return {"valid": not missing and not unknown, "missing": missing, "unknown": unknown, "feature_count": len(self.features)}

    def write(self, path: str | Path) -> None:
        payload = {"scenario": self.scenario, "version": self.version, "features": {name: asdict(spec) for name, spec in sorted(self.features.items())}}
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


EXPLICIT_METADATA_KEYS = {
    "interaction_cpu_build_seconds", "interaction_gpu_compute_seconds", "interaction_transfer_seconds",
    "interaction_backend", "interaction_backend_reason", "interaction_gpu_available", "feature_schema_id",
    "schema_version", "topology_valid", "persistence_valid", "status", "path", "mode", "source_path",
    "hodge_method", "hodge_converged", "hodge_residual", "hodge_estimation_error", "feature_count",
    "support_solver_seconds", "support_cpp_seconds", "support_gpu_seconds", "support_pipeline_seconds",
    "support_cpp_error", "support_gpu_error", "support_solver_backend", "support_active_set_solver",
    "support_cpp_available", "support_gpu_available", "support_cpp_attempted", "support_gpu_attempted",
    "support_solver_failures", "support_fallbacks", "support_certificate_failures", "support_unverified",
}

# These values belong to labels/cohort bookkeeping, never to a structural
# predictor.  Keeping them here prevents a flattened input row from leaking
# iRMSD or candidate identity into M0/M1/M2.
LABEL_OR_COHORT_KEYS = {
    "target_id", "candidate_id", "quality", "irmsd", "fnat", "contact_recovery",
    "dockq", "capri", "label", "fold", "split", "dataset", "scenario",
}

SCENARIO_REQUIRED_KEYS = {
    "patch_count", "patch_directed_edge_count", "path_count_k0", "path_count_k1",
    "path_count_k2", "path_count_k3", "pathchem_count_k0", "pathchem_count_k1",
    "pathchem_count_k2", "pathchem_count_k3", "tau_2_mean", "tau_3_mean",
    "tau_4_mean", "eta_3_mean", "eta_4_mean", "delta_tau_2_mean",
    "delta_tau_3_mean", "delta_tau_4_mean",
}

# ``*_sparse`` scenarios are the GSPH-S task-adapted variants: they may run
# with ``compute_eta=False`` and a reduced max_path_order, so the eta
# statistics (and, when order<3, the k3 path counts which are legitimately
# zero) are not part of their contract.  Sparse rows are still required to
# carry every other structural key; missing values are never zero-filled.
_SPARSE_OPTIONAL_KEYS = {"eta_3_mean", "eta_4_mean"}


def required_keys_for(scenario: str):
    keys = set(SCENARIO_REQUIRED_KEYS)
    if str(scenario).endswith("_sparse"):
        keys -= _SPARSE_OPTIONAL_KEYS
    return keys


def load_feature_manifest(scenario: str, root: str | Path | None = None) -> FeatureSchema:
    """Load the checked-in exact structural manifest for ``scenario``."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    path = base / "schemas" / f"{scenario}_v6_features.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scenario") != scenario or payload.get("version") != "gsph6.v1":
        raise ValueError(f"manifest scenario/version mismatch: {path}")
    specs = {}
    for name, raw in (payload.get("features") or {}).items():
        specs[str(name)] = FeatureSpec(
            name=str(raw.get("name", name)), dtype=str(raw.get("dtype", "float64")),
            unit=str(raw.get("unit", "dimensionless")), block=str(raw.get("block", "unknown")),
            definition=str(raw.get("definition", "")),
            structural_zero_allowed=bool(raw.get("structural_zero_allowed", True)),
            missing_allowed=bool(raw.get("missing_allowed", False)),
        )
    return FeatureSchema(scenario=scenario, version=str(payload["version"]), features=specs)


def validate_feature_record(values: Mapping[str, object], scenario: str, schema: FeatureSchema | None = None) -> Dict[str, object]:
    """Validate the fixed structural core before a row can enter evaluation."""
    numeric_bad = sorted(str(key) for key, value in values.items() if not isinstance(value, (int, float)) or not math.isfinite(float(value)))
    missing = sorted(key for key in required_keys_for(scenario) if key not in values)
    leaked = sorted(key for key in values if key in LABEL_OR_COHORT_KEYS or key in EXPLICIT_METADATA_KEYS)
    exact = schema.validate(values) if schema is not None else {"valid": True, "missing": [], "unknown": []}
    return {"valid": not numeric_bad and not missing and not leaked and bool(exact.get("valid", False)), "missing": sorted(set(missing) | set(exact.get("missing", []))), "unknown": sorted(set(exact.get("unknown", []))), "non_finite_or_non_numeric": numeric_bad, "leaked_keys": leaked, "scenario": str(scenario), "schema_id": f"gsph6.v1:{scenario}", "manifest_feature_count": len(schema.features) if schema is not None else None}


def compact_features_v6(record: Mapping[str, object], schema: FeatureSchema | None = None) -> Dict[str, float]:
    """Keep structural features; do not filter by the substring ``path``."""
    raw = dict(record.get("features", record) or {})
    out: Dict[str, float] = {}
    for key, value in raw.items():
        if key in EXPLICIT_METADATA_KEYS or key in LABEL_OR_COHORT_KEYS or not isinstance(value, (int, float)):
            continue
        if not (value == value and abs(float(value)) != float("inf")):
            continue
        out[str(key)] = float(value)
    if schema is not None:
        out = {name: float(out.get(name, 0.0)) for name in schema.features}
    return out


def schema_from_features(scenario: str, values: Mapping[str, object], version: str = "gsph6.v1") -> FeatureSchema:
    features = {}
    for name in sorted(k for k, v in values.items() if isinstance(v, (int, float)) and k not in EXPLICIT_METADATA_KEYS and k not in LABEL_OR_COHORT_KEYS):
        block = name.split("_", 1)[0]
        features[name] = FeatureSpec(name=name, block=block, definition="fixed GSPH6 structural feature")
    return FeatureSchema(scenario=scenario, version=version, features=features)


__all__ = ["FeatureSpec", "FeatureSchema", "compact_features_v6", "schema_from_features", "load_feature_manifest", "LABEL_OR_COHORT_KEYS", "validate_feature_record", "SCENARIO_REQUIRED_KEYS"]
