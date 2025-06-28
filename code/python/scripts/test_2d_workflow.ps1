# Test script for 2D calculation and plotting workflow (PowerShell)
#
# This script tests the new feed-forward workflow where:
# 1. calc_2D_datas.py outputs both data_path and info_path
# 2. plot_2D_datas.py accepts these paths as --data-path and --info-path arguments
# 3. Figures are automatically saved in matching directory structure
#
# Usage: .\test_2d_workflow.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TESTING 2D CALCULATION AND PLOTTING WORKFLOW" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python environment
Write-Host "Checking Python environment..." -ForegroundColor Yellow
if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} elseif ($env:CONDA_DEFAULT_ENV) {
    Write-Host "✅ Conda environment detected: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠️  No virtual environment detected, using system Python" -ForegroundColor Yellow
}

try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor White
} catch {
    Write-Host "❌ Python not found" -ForegroundColor Red
    exit 1
}

# Set matplotlib backend to non-interactive
$env:MPLBACKEND = "Agg"

Write-Host ""
Write-Host "=== STEP 1: Testing calculation phase ===" -ForegroundColor Yellow
Write-Host "Running: python calc_2D_datas.py" -ForegroundColor White

# Run calc_2D_datas.py and capture output
try {
    $calcOutput = python calc_2D_datas.py 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "calc_2D_datas.py failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}

# Extract data and info paths from the output
$dataPath = ($calcOutput | Select-String "Data file: (.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
$infoPath = ($calcOutput | Select-String "Info file: (.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()

# Validate that both paths were captured
if (-not $dataPath) {
    Write-Host "❌ Error: No data file path returned from calc_2D_datas.py" -ForegroundColor Red
    Write-Host "Full output:" -ForegroundColor Red
    Write-Host $calcOutput -ForegroundColor Gray
    exit 1
}

if (-not $infoPath) {
    Write-Host "❌ Error: No info file path returned from calc_2D_datas.py" -ForegroundColor Red
    Write-Host "Full output:" -ForegroundColor Red
    Write-Host $calcOutput -ForegroundColor Gray
    exit 1
}

Write-Host "✅ DATA_PATH: $dataPath" -ForegroundColor Green
Write-Host "✅ INFO_PATH: $infoPath" -ForegroundColor Green

# Verify files exist
if (-not (Test-Path $dataPath)) {
    Write-Host "❌ Error: Data file does not exist: $dataPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $infoPath)) {
    Write-Host "❌ Error: Info file does not exist: $infoPath" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Both data files verified to exist" -ForegroundColor Green

Write-Host ""
Write-Host "=== STEP 2: Testing plotting phase ===" -ForegroundColor Yellow
Write-Host "Running: python plot_2D_datas.py --data-path `"$dataPath`" --info-path `"$infoPath`"" -ForegroundColor White

# Run plot_2D_datas.py with the captured file paths
try {
    python plot_2D_datas.py --data-path "$dataPath" --info-path "$infoPath"
    if ($LASTEXITCODE -ne 0) {
        throw "plot_2D_datas.py failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Plotting completed successfully" -ForegroundColor Green

Write-Host ""
Write-Host "=== WORKFLOW TEST COMPLETED ===" -ForegroundColor Cyan
Write-Host "✅ Successfully tested the complete calculation → plotting workflow" -ForegroundColor Green
Write-Host "✅ Data files: $dataPath" -ForegroundColor Green
Write-Host "✅ Info files: $infoPath" -ForegroundColor Green
Write-Host "✅ Plots should be saved in the figures directory" -ForegroundColor Green
