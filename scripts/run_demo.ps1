$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# Nạp .env (KEY=VALUE, bỏ qua dòng trống và comment)
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#=\s][^=]*)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
} else {
    Write-Host "Không có .env (dùng giá trị mặc định / auto-discover)." -ForegroundColor Yellow
}

python -X utf8 -m release_control.cli demo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Mở dashboard:" -ForegroundColor Yellow
Write-Host "  python -X utf8 -m uvicorn dashboard.app:app --port 8765"
