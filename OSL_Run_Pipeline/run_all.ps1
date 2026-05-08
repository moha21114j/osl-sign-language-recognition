$ErrorActionPreference = 'Stop'

Write-Host 'Step 1/3: Train OSL words (ISLR)' -ForegroundColor Cyan
python "$PSScriptRoot/run_osl_training.py"

Write-Host 'Step 2/3: Prepare OSL sentences dataset' -ForegroundColor Cyan
python "$PSScriptRoot/prepare_osl_sentences_dataset.py"

Write-Host 'Step 3/3: Train OSL sentences (SLT)' -ForegroundColor Cyan
python "$PSScriptRoot/run_osl_sentences_training.py"

Write-Host 'All steps completed.' -ForegroundColor Green
