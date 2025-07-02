# =============================================================================
# 1D Spectroscopy Workflow Test Script (PowerShell version)
# =============================================================================
# This script runs the complete 1D spectroscopy workflow:
# 1. Generate data with calc_1D_datas.py
# 2. Automatically plot the results with plot_1D_datas.py
#
# Usage: .\test_1d_workflow.ps1
# =============================================================================

$ErrorActionPreference = "Stop"  # Exit on any error

Write-Host "Starting 1D Spectroscopy Workflow Test" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# =============================
# STEP 1: RUN SIMULATION
# =============================
Write-Host ""
Write-Host "Step 1: Running 1D simulation..." -ForegroundColor Yellow
Write-Host ""

# Run the calculation and capture the output
Write-Host "Executing: python calc_1D_datas.py"
try {
    python calc_1D_datas.py > calc_output.log 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Simulation completed successfully!" -ForegroundColor Green
    }
    else {
        throw "Simulation failed with exit code $LASTEXITCODE"
    }
}
catch {
    Write-Host "Simulation failed! Check calc_output.log for details." -ForegroundColor Red
    exit 1
}

# =============================
# STEP 2: EXTRACT PATHS FROM OUTPUT
# =============================
Write-Host ""
Write-Host "Step 2: Extracting file paths..." -ForegroundColor Yellow

# Extract the plotting command from the output
$plotCmd = Get-Content calc_output.log | Select-String "python plot_1D_datas.py" | Select-Object -Last 1

if (-not $plotCmd) {
    Write-Host "Could not find plotting command in output!" -ForegroundColor Red
    Write-Host "Last few lines of calc_output.log:"
    Get-Content calc_output.log | Select-Object -Last 10
    exit 1
}

Write-Host "Found plotting command: $($plotCmd.Line)"

# =============================
# STEP 3: RUN PLOTTING
# =============================
Write-Host ""
Write-Host "Step 3: Running plotting script..." -ForegroundColor Yellow
Write-Host "Executing: $($plotCmd.Line)"

# Execute the plotting command
try {
    Invoke-Expression $plotCmd.Line
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Plotting completed successfully!" -ForegroundColor Green
    }
    else {
        throw "Plotting failed with exit code $LASTEXITCODE"
    }
}
catch {
    Write-Host "Plotting failed!" -ForegroundColor Red
    exit 1
}

# =============================
# WORKFLOW COMPLETE
# =============================
Write-Host ""
Write-Host "1D Spectroscopy Workflow Completed Successfully!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Generated files:"
Get-Content calc_output.log | Select-String "Data file:" | Select-Object -Last 1
Get-Content calc_output.log | Select-String "Info file:" | Select-Object -Last 1
Write-Host ""
Write-Host "Plots saved to: figures/1d_spectroscopy/subdir/"
Write-Host ""
Write-Host "Full simulation log available in: calc_output.log"
Write-Host "=================================================" -ForegroundColor Green

# Clean up
Remove-Item calc_output.log -ErrorAction SilentlyContinue