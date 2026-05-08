$ErrorActionPreference = 'Stop'

Write-Host 'Running OSL words training (ISLR)...' -ForegroundColor Cyan
python "$PSScriptRoot/run_osl_training.py"
