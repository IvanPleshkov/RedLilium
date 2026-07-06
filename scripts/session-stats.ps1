<#
.SYNOPSIS
  Token/agent statistics for a Claude Code session (Windows/PowerShell).

.DESCRIPTION
  Reads the on-disk session transcript Claude Code writes under
  ~/.claude/projects/<slug>/ and reports:
    - main-loop token usage per model (the tier that owned the session)
    - subagents rolled up by tier, with an optional per-agent breakdown
    - subagent launches by requested type

.PARAMETER SessionId
  Session uuid. Defaults to the most recently modified transcript.

.PARAMETER Since
  Only count records at/after this ISO date (e.g. 2026-07-06) — use it to
  scope a long/compacted transcript to a single day.

.PARAMETER PerAgent
  Also print one row per subagent.

.EXAMPLE
  .\scripts\session-stats.ps1 -Since 2026-07-06 -PerAgent
#>
param(
  [string]$SessionId = "",
  [string]$Since = "",
  [switch]$PerAgent
)
$ErrorActionPreference = "Stop"

function Fmt([double]$n) { '{0:N0}' -f $n }
function NZ($v) { if ($null -eq $v) { 0 } else { $v } }   # null -> 0 (PS 5.1-safe)

# Resolve the project transcript directory from the repo path (Claude encodes
# the absolute path with path separators replaced by '-'; on Windows the drive
# colon is folded too). If the derived name is absent we fall back to matching
# by the repo's leaf folder name, since the exact encoding can vary.
$root = (& git rev-parse --show-toplevel 2>$null)
if (-not $root) { $root = (Get-Location).Path }
$projectsRoot = if ($env:CLAUDE_PROJECTS) { $env:CLAUDE_PROJECTS } else { Join-Path $HOME ".claude/projects" }
if (-not (Test-Path $projectsRoot)) { Write-Error "no transcripts root at $projectsRoot"; exit 1 }

$slug = ($root -replace '[\\/:]', '-')
$proj = Join-Path $projectsRoot $slug
if (-not (Test-Path $proj)) {
  $leaf = Split-Path $root -Leaf
  $cand = Get-ChildItem $projectsRoot -Directory | Where-Object { $_.Name -like "*-$leaf" }
  if ($cand.Count -eq 1) { $proj = $cand[0].FullName }
  else { Write-Error "could not resolve project dir under $projectsRoot (tried '$slug')"; exit 1 }
}

# Pick the session file.
if ($SessionId) {
  $f = Join-Path $proj "$SessionId.jsonl"
} else {
  $newest = Get-ChildItem $proj -Filter *.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $f = $newest.FullName; $SessionId = $newest.BaseName
}
if (-not (Test-Path $f)) { Write-Error "session file not found: $f"; exit 1 }

function PassSince($ts) { -not $Since -or ($ts -ge $Since) }

$item = Get-Item $f
$sizeMB = [math]::Round($item.Length / 1MB, 1)
Write-Host "RedLilium - session stats"
Write-Host "  project : $root"
Write-Host ("  session : {0}  ({1} MB, {2:yyyy-MM-dd HH:mm})" -f $SessionId, $sizeMB, $item.LastWriteTime)
if ($Since) { Write-Host "  since   : $Since" }
Write-Host ""

# --- Main loop, by model ---
$byModel = @{}
$launches = @{}
foreach ($line in [System.IO.File]::ReadLines($f)) {
  if (-not $line) { continue }
  try { $r = $line | ConvertFrom-Json } catch { continue }
  if ($r.type -ne 'assistant') { continue }
  if (-not (PassSince $r.timestamp)) { continue }
  if ($r.message.usage) {
    $m = if ($r.message.model) { $r.message.model } else { '?' }
    if ($m -ne '<synthetic>') {
      if (-not $byModel[$m]) { $byModel[$m] = [pscustomobject]@{ msgs=0; o=0.0; i=0.0; cc=0.0; cr=0.0 } }
      $u = $r.message.usage
      $byModel[$m].msgs++
      $byModel[$m].o  += [double](NZ $u.output_tokens)
      $byModel[$m].i  += [double](NZ $u.input_tokens)
      $byModel[$m].cc += [double](NZ $u.cache_creation_input_tokens)
      $byModel[$m].cr += [double](NZ $u.cache_read_input_tokens)
    }
  }
  foreach ($c in @($r.message.content)) {
    if ($c.type -eq 'tool_use' -and ($c.name -eq 'Task' -or $c.name -eq 'Agent')) {
      $t = if ($c.input.subagent_type) { $c.input.subagent_type } else { 'unspecified' }
      $launches[$t] = [int]$launches[$t] + 1
    }
  }
}
Write-Host "Main loop - by model (the tier that owned the session):"
$byModel.GetEnumerator() | Sort-Object { $_.Value.o } -Descending | ForEach-Object {
  $v = $_.Value
  "  {0,-24} msgs={1,-6} output={2,-13} input={3,-11} cache_creation={4,-12} cache_read={5}" -f `
    $_.Key, $v.msgs, (Fmt $v.o), (Fmt $v.i), (Fmt $v.cc), (Fmt $v.cr)
}
Write-Host ""

Write-Host "Subagent launches - by requested type:"
$launches.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
  "  {0,-22} x{1}" -f $_.Key, $_.Value
}
Write-Host ""

# --- Subagents, from their own transcripts ---
$sub = Join-Path $proj "$SessionId/subagents"
$agentFiles = @()
if (Test-Path $sub) { $agentFiles = Get-ChildItem $sub -Filter "agent-*.jsonl" -ErrorAction SilentlyContinue }
if ($agentFiles.Count -gt 0) {
  $rows = foreach ($af in $agentFiles) {
    $id = $af.BaseName
    $metaPath = Join-Path $sub "$id.meta.json"
    $type = '?'
    if (Test-Path $metaPath) { try { $type = (Get-Content $metaPath -Raw | ConvertFrom-Json).agentType } catch {} }
    $model = '?'; $msgs = 0; $out = 0.0
    foreach ($line in [System.IO.File]::ReadLines($af.FullName)) {
      if (-not $line) { continue }
      try { $r = $line | ConvertFrom-Json } catch { continue }
      if (-not $r.message.usage) { continue }
      if (-not (PassSince $r.timestamp)) { continue }
      if ($model -eq '?' -and $r.message.model) { $model = $r.message.model }
      $msgs++; $out += [double](NZ $r.message.usage.output_tokens)
    }
    if ($msgs -gt 0) {
      [pscustomobject]@{ model=$model; type=$type; id=($id -replace '^agent-',''); msgs=$msgs; out=$out }
    }
  }
  Write-Host "Subagents - by tier:"
  $rows | Group-Object model | Sort-Object { ($_.Group | Measure-Object out -Sum).Sum } -Descending | ForEach-Object {
    $sum = ($_.Group | Measure-Object out -Sum).Sum
    "  {0,-24} agents={1,-4} output={2}" -f $_.Name, $_.Count, (Fmt $sum)
  }
  if ($PerAgent) {
    Write-Host ""
    Write-Host "Subagents - per agent (tier / type / msgs / output):"
    $rows | Sort-Object out -Descending | ForEach-Object {
      "  {0,-24} {1,-16} {2,-8} msgs={3,-4} output={4}" -f $_.model, $_.type, $_.id.Substring(0, [Math]::Min(8,$_.id.Length)), $_.msgs, (Fmt $_.out)
    }
  }
} else {
  Write-Host "Subagents - none recorded for this session."
}
Write-Host ""
Write-Host "Note: a long-lived/compacted transcript accumulates many turns across"
Write-Host "days and tiers. Scope to one day with -Since YYYY-MM-DD. 'output' is the"
Write-Host "cost-relevant axis for tier arbitrage (cache_read is near-free)."
