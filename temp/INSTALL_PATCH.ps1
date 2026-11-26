param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot
)
$ErrorActionPreference = "Stop"
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& py -3.12 (Join-Path $PatchRoot "apply_patch.py") --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Patch application failed" }
Write-Host "Patch files copied. Install dependencies and spaCy language models as described in PATCH_README_RU.md."
