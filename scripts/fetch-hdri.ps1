# Fetch the pinned source HDRI for the IBL bake (#137) into the repo-local
# `.hdri\` directory (gitignored). Windows twin of scripts/fetch-hdri.sh —
# see that file for the pinning rationale.
$ErrorActionPreference = "Stop"

$HdriName = "spruit_sunrise_2k.hdr"
$HdriUrl = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/$HdriName"

$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root ".hdri"
$File = Join-Path $Dest $HdriName

if (Test-Path $File) {
    Write-Host "$HdriName already present in $Dest - nothing to do."
    exit 0
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Write-Host "Downloading $HdriUrl"
Invoke-WebRequest -Uri $HdriUrl -OutFile $File

$Digest = (Get-FileHash -Algorithm SHA256 $File).Hash.ToLower()
Write-Host "sha256: $Digest"
Write-Host "$HdriName provisioned in $Dest"
