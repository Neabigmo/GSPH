# EXP-04: Mathematical Audits (solver stress + perturbation stability)

- **Purpose**: Certify the support solver and verify numerical stability
  of tau under bounded coordinate perturbation.
- **Data**: 500 adversarial patch configurations (coincident, collinear,
  nested, extreme separation/aspect ratio); 8 structures x ~300
  canonical patch sets x 4 noise levels.
- **Method**: `solver_stress.py` (definition-direct verification);
  `exp04_perturb.py` (patch dumps rebuilt, atoms perturbed
  N(0, sigma^2 I), tau recomputed + certified); `p0_audits.py`,
  `p0_chain_audit.py` (GLMY boundary^2=0, chain maps).
- **Key results**:
  - 500/500 stress cases certified, deterministic, witness-consistent.
  - Perturbation: mean |d tau| = 0.025/0.050/0.102/0.277 A for
    sigma = 0.05/0.10/0.20/0.50 A; max 0.758 A; zero certificate
    failures -- near-linear Lipschitz-style response.
  - GLMY chain-complex audits 60/60 pass.
- **Status**: DONE.
- **Artifacts**: `results/tables/solver_stress.json`,
  `path_drift.csv`, `patch_drift.csv`, `p0_audits.json`,
  `p0_chain_audit.json`.
