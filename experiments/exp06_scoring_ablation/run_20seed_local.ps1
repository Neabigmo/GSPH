# EXP-06: 20-seed grouped CV driver (local; feature parquets in GSPH_H3/_feat)
$ErrorActionPreference = "Continue"
Set-Location "H:\2026try\11.3GSPH\GSPH_H3\ablation"
$py = "F:\anaconda3\python.exe"
foreach ($task in @("rna", "pd_bound", "pd_unbound")) {
  foreach ($seed in 0..19) {
    & $py run_ablation.py --task $task --seed $seed --tag "_s$seed" |
      Select-Object -Last 2
    Write-Output "[20seed] $task s$seed done"
  }
}
& $py "H:\2026try\11.3GSPH\cspph-molecular\experiments\exp06_scoring_ablation\merge_oof.py"
Write-Output "[20seed] ALL DONE"
