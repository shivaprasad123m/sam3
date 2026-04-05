# Run SAM3 inference with Hugging Face token setup
# Usage: Open PowerShell as admin/user, then run:
#   .\run_inference.ps1

# 1. Set HF token
$Token = ""
setx HUGGINGFACE_TOKEN $Token

# 2. Optional login method
Write-Host "Now run: huggingface-cli login and paste token if not already logged in."
Write-Host "Press Enter to continue..."
[Console]::ReadLine() | Out-Null

# 3. Activate venv
Push-Location "C:\Shiva\Projects\SAM3\sam3"
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "Virtual environment activation script not found; ensure .venv is created." -ForegroundColor Yellow
}

# 4. Run inference script
Write-Host "Running inference_script.py..."
& "c:\Shiva\Projects\SAM3\sam3\.venv\Scripts\python.exe" "inference_script.py"

# 5. List outputs
Write-Host "Inference output files:"
Get-ChildItem -Path ".\inference_results" -File | Select-Object Name, Length

Pop-Location
