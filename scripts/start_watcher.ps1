# scripts/start_watcher.ps1
# Start the GS_PIPELINE watcher in a reproducible environment (venv, correct cwd, logs).
# Designed to be run from Task Scheduler.

# --- config ---
$RepoRoot = "D:\GS-PIPELINE-PRO"
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
$LogDir = Join-Path $RepoRoot "logs"
$LogFile = Join-Path $LogDir ("watcher_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
# ---

# Ensure log dir exists
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Make sure script stops on errors
$ErrorActionPreference = "Stop"

# Set working dir
Set-Location -Path $RepoRoot

# Activate venv
if (Test-Path $VenvActivate) {
    try {
        . $VenvActivate
    } catch {
        Write-Output "Failed to activate venv: $_" | Out-File -FilePath $LogFile -Append
        throw
    }
} else {
    "WARNING: venv activate script not found at $VenvActivate" | Out-File -FilePath $LogFile -Append
}

# Optional: add venv Scripts to PATH (redundant if Activate worked)
$venvScripts = Join-Path $RepoRoot ".venv\Scripts"
if (Test-Path $venvScripts) {
    $env:PATH = $venvScripts + ";" + $env:PATH
}

# Run watcher in foreground, log stdout+stderr
Write-Output "Starting watcher at $(Get-Date)" | Out-File -FilePath $LogFile -Append
# Ensure the process keeps running under Task Scheduler; do not detach
& python watcher.py *>&1 | Tee-Object -FilePath $LogFile -Append
$exitCode = $LASTEXITCODE
Write-Output "Watcher exited with code $exitCode at $(Get-Date)" | Out-File -FilePath $LogFile -Append
exit $exitCode
