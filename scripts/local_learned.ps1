param([int]$PerCharge = 500, [int]$Workers = 4)
$ErrorActionPreference = 'Stop'
if ($PerCharge -lt 10 -or $PerCharge -gt 500) { throw 'PerCharge must be 10..500 for this bounded first experiment.' }
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path -LiteralPath 'data/learned_v1/pool/manifest.json')) {
    python -m egc.learned pool --per-charge $PerCharge
    if ($LASTEXITCODE -ne 0) { throw 'Pool preparation failed; inspect output.' }
} else {
    $poolConfig = Get-Content -LiteralPath 'data/learned_v1/pool/manifest.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($poolConfig.per_charge_cap -ne $PerCharge) { throw 'Existing pool has a different PerCharge; use explicit module commands with a fresh path.' }
}
python -m egc.learned annotate --workers $Workers --limit 6000 --ask-key
if ($LASTEXITCODE -ne 0) { throw 'Annotation failed; cache is preserved. Do not delete pending requests.' }
if (-not (Test-Path -LiteralPath 'data/learned_v1/ready/experiment.zip')) {
    python -m egc.learned prepare
    if ($LASTEXITCODE -ne 0) { throw 'Packaging failed; inspect incomplete annotations/pending requests.' }
}
Write-Host 'Send data/learned_v1/ready/experiment.zip privately to the server. No dataset/API cache goes to GitHub.'
Write-Host 'Keep data/learned_v1/annotations/summary.json and data/learned_v1/ready/teacher_review.jsonl for review.'
