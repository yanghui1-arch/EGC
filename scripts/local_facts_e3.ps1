param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $egcArgs = @('-m', 'egc', 'facts-run', '--input', 'data/facts_e2_v1/new.jsonl',
        '--profile', 'configs/evidence_profile_e3.json', '--protocol', 'configs/facts_e3_protocol.json',
        '--output-dir', 'data/facts_e3_v1/new_run', '--modes', 'flat_e3', 'bound_e3',
        '--limit', '120', '--max-tokens', '4096')
    & python @egcArgs --dry-run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($DryRun) { exit 0 }
    & python @egcArgs --ask-key
    $egcRunExit = $LASTEXITCODE
    if ($egcRunExit -notin @(0, 3)) { exit $egcRunExit }
    & python -m egc facts-report --input data/facts_e2_v1/new.jsonl --run-dir data/facts_e3_v1/new_run --output data/facts_e3_v1/new_run/comparison.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($egcRunExit -eq 3) {
        Write-Warning 'Some requests failed. Report saved; stop here and return the results. No automatic retries.'
    }
    exit $egcRunExit
} finally {
    Pop-Location
}
