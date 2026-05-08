$ErrorActionPreference = 'Stop'

Write-Host 'Running OSL sentences training (SLT)...' -ForegroundColor Cyan
python "$PSScriptRoot/run_osl_sentences_training.py"
