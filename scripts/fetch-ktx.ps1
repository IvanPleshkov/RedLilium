# Provision the pinned KTX-Software CLI (`ktx`) into the repo-local `.ktx\`
# directory (gitignored). Windows twin of scripts/fetch-ktx.sh — see that file
# for the pinning rationale. Runs the official installer silently into .ktx\.
#
#   .\scripts\fetch-ktx.ps1
#   .\.ktx\bin\ktx.exe --version
$ErrorActionPreference = "Stop"

$KtxVersion = "4.4.2"

$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root ".ktx"
$KtxExe = Join-Path $Dest "bin\ktx.exe"

if ((Test-Path $KtxExe) -and ((& $KtxExe --version) -match $KtxVersion)) {
    Write-Host "ktx $KtxVersion already present in $Dest - nothing to do."
    exit 0
}

$Arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
$Asset = "KTX-Software-$KtxVersion-Windows-$Arch.exe"
$Url = "https://github.com/KhronosGroup/KTX-Software/releases/download/v$KtxVersion/$Asset"
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $Tmp | Out-Null

try {
    $Installer = Join-Path $Tmp $Asset
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Installer

    $Digest = (Get-FileHash -Algorithm SHA256 $Installer).Hash.ToLower()
    Write-Host "sha256: $Digest"

    if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
    # The release installer supports silent installation into a target dir.
    Write-Host "Installing into $Dest"
    Start-Process -FilePath $Installer -ArgumentList "/S", "/D=$Dest" -Wait -NoNewWindow

    if (-not (Test-Path $KtxExe)) {
        throw "installer did not produce $KtxExe - install KTX-Software $KtxVersion manually into $Dest"
    }
    & $KtxExe --version
    Write-Host "ktx $KtxVersion provisioned in $Dest"
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
