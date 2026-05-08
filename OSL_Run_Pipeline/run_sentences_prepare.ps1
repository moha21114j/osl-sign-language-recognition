$ErrorActionPreference = 'Stop'

Write-Host 'Preparing OSL sentences dataset for Uni-Sign...' -ForegroundColor Cyan
python "$PSScriptRoot/prepare_osl_sentences_dataset.py"
