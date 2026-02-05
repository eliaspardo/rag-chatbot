<#
.SYNOPSIS
  Apply the MLflow compare-parents patch in a Windows-friendly way.

.DESCRIPTION
  This script copies a prepared MLflow server __init__.py template into the active venv's
  site-packages path so the custom /compare-parents route is available.

  It is intentionally verbose and heavily commented so failures are easy to diagnose.

.USAGE
  PS> .\scripts\apply_mlflow_compare_patch.ps1

  Optional overrides:
    $Env:VIRTUAL_ENV      -> path to the active venv
    $Env:MLFLOW_SERVER_INIT -> full path to target __init__.py (bypasses discovery)
#>

# Fail fast if any command errors.
$ErrorActionPreference = "Stop"

# Resolve repo root based on this script's location.
$RootDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PatchFile = Join-Path $RootDir "tools\mlflow_patches\patches\mlflow-compare-parents.patch"
$TemplateFile = Join-Path $RootDir "tools\mlflow_patches\patches\mlflow_server_init.py"
$BackupDir = Join-Path $RootDir "tools\mlflow_patches\backup"

# Determine venv path.
# If VIRTUAL_ENV is set (most reliable), use it. Otherwise, fall back to .venv.
$VenvPath = $Env:VIRTUAL_ENV
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
  $VenvPath = Join-Path $RootDir ".venv"
}

# Allow direct override when discovery is tricky.
$TargetInit = $Env:MLFLOW_SERVER_INIT

# Locate Python inside the venv. On Windows, it's usually in Scripts\python.exe.
$PythonBin = ""
$Candidate = Join-Path $VenvPath "Scripts\python.exe"
if (Test-Path $Candidate) {
  $PythonBin = $Candidate
}

# If we still don't have a Python path, try to use 'python' from PATH.
# This is less reliable but better than failing without a hint.
if ([string]::IsNullOrWhiteSpace($PythonBin)) {
  $PythonOnPath = Get-Command python -ErrorAction SilentlyContinue
  if ($PythonOnPath) {
    $PythonBin = $PythonOnPath.Path
  }
}

# Resolve site-packages using Python itself (most accurate across environments).
if ([string]::IsNullOrWhiteSpace($TargetInit)) {
  if ([string]::IsNullOrWhiteSpace($PythonBin)) {
    throw "Unable to locate Python. Activate your venv or set VIRTUAL_ENV."
  }
  $SitePackages = & $PythonBin -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  if ([string]::IsNullOrWhiteSpace($SitePackages)) {
    throw "Python did not return a site-packages path."
  }
  $TargetInit = Join-Path $SitePackages "mlflow\server\__init__.py"
}

# Ensure we have at least one source artifact to apply.
if (-not (Test-Path $TemplateFile) -and -not (Test-Path $PatchFile)) {
  throw "Missing patch/template. Expected $TemplateFile or $PatchFile."
}

# Ensure parent directory exists even if __init__.py does not.
$TargetDir = Split-Path -Parent $TargetInit
if (-not (Test-Path $TargetDir)) {
  throw "Target directory does not exist: $TargetDir. Is MLflow installed in this venv?"
}

# Back up any existing __init__.py (if it exists).
if (-not (Test-Path $BackupDir)) {
  New-Item -ItemType Directory -Path $BackupDir | Out-Null
}
if (Test-Path $TargetInit) {
  $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $BackupFile = Join-Path $BackupDir "__init__.py.$Timestamp"
  Copy-Item $TargetInit $BackupFile
}

# Apply the template directly (preferred). This avoids patch tool issues on Windows.
if (Test-Path $TemplateFile) {
  Copy-Item $TemplateFile $TargetInit -Force
  Write-Host "Patched MLflow server __init__.py via template:" $TargetInit
  exit 0
}

# Fallback: apply patch (requires a patch tool installed on PATH, e.g., git-bash or GnuWin32).
# This is less reliable on Windows, so template is strongly preferred.
$PatchTool = Get-Command patch -ErrorAction SilentlyContinue
if (-not $PatchTool) {
  throw "Template missing and 'patch' tool not found. Install patch or use the template approach."
}

# Remove target then patch (the patch is a full-file patch).
Remove-Item $TargetInit -Force -ErrorAction SilentlyContinue
& $PatchTool.Path -p0 -i $PatchFile
Write-Host "Patched MLflow server __init__.py via patch tool." 
