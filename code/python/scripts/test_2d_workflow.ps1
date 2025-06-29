# =============================================================================
# 2D Spectroscopy Workflow Test Script (PowerShell version)
# =============================================================================
# This script runs the complete 2D spectroscopy workflow:
# 1. Generate data with calc_2D_datas.py
# 2. Automatically plot the results with plot_2D_datas.py
#
# Usage: .\test_2d_workflow.ps1
# =============================================================================

$ErrorActionPreference = "Stop"  # Exit on any error

Write-Host "Starting 2D Spectroscopy Workflow Test" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# =============================
# STEP 1: RUN SIMULATION
# =============================
Write-Host ""
Write-Host "Step 1: Running 2D simulation..." -ForegroundColor Yellow
Write-Host ""

# Run the calculation and capture the output
Write-Host "Executing: python calc_2D_datas.py"
try {
    python calc_2D_datas.py > calc_output.log 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Simulation completed successfully!" -ForegroundColor Green
    }
    else {
        throw "Simulation failed with exit code $LASTEXITCODE"
    }
}
catch {
    Write-Host "Simulation failed! Check calc_output.log for details." -ForegroundColor Red
    Get-Content calc_output.log | Select-Object -Last 20
    exit 1
}

# =============================
# STEP 2: EXTRACT PATHS FROM OUTPUT
# =============================
Write-Host ""
Write-Host "Step 2: Extracting file paths..." -ForegroundColor Yellow

# Look for the plotting command in the output
$plotCmdLine = Get-Content calc_output.log | Where-Object { $_ -match "python plot_2D_datas.py" } | Select-Object -Last 1

if (-not $plotCmdLine) {
    Write-Host "Could not find plotting command in output!" -ForegroundColor Red
    Write-Host "Full calc_output.log content:"
    Get-Content calc_output.log
    exit 1
}

Write-Host "Found plotting command: $plotCmdLine"

# Extract the arguments from the command line
if ($plotCmdLine -match 'python plot_2D_datas.py --data-path "([^"]+)" --info-path "([^"]+)"') {
    $dataPath = $matches[1]
    $infoPath = $matches[2]
    $plotCommand = "python plot_2D_datas.py --data-path `"$dataPath`" --info-path `"$infoPath`""
}
else {
    Write-Host "Could not parse plotting command arguments!" -ForegroundColor Red
    Write-Host "Command line: $plotCmdLine"
    exit 1
}

# =============================
# STEP 3: RUN PLOTTING
# =============================
Write-Host ""
Write-Host "Step 3: Running plotting script..." -ForegroundColor Yellow
Write-Host "Executing: $plotCommand"

# Execute the plotting command
try {
    Invoke-Expression $plotCommand
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
    Write-Host "Error: $_"
    exit 1
}

# =============================
# WORKFLOW COMPLETE
# =============================
Write-Host ""
Write-Host "2D Spectroscopy Workflow Completed Successfully!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Generated files:"
Get-Content calc_output.log | Select-String "Data file:" | Select-Object -Last 1
Get-Content calc_output.log | Select-String "Info file:" | Select-Object -Last 1
Write-Host ""
Write-Host "Plots saved to: figures/figures_from_python/2d_spectroscopy/"
Write-Host ""
Write-Host "Full simulation log available in: calc_output.log"
Write-Host "=================================================" -ForegroundColor Green

# Clean up
Remove-Item calc_output.log -ErrorAction SilentlyContinue