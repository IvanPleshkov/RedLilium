#!/usr/bin/env bash
#
# session-stats.sh — token/agent statistics for a Claude Code session.
#
# Reads the on-disk session transcript that Claude Code writes under
# ~/.claude/projects/<slug>/ and reports:
#   - main-loop token usage per model (the tier that owned the session)
#   - subagents rolled up by tier, with a per-agent breakdown
#   - subagent launches by requested type
#
# Usage:
#   scripts/session-stats.sh [SESSION_ID] [--since YYYY-MM-DD] [--per-agent]
#
#   SESSION_ID    session uuid (defaults to the most recently modified one)
#   --since DATE  only count records at/after DATE (ISO, e.g. 2026-07-06) —
#                 use this to scope a long/compacted transcript to one day
#   --per-agent   also print one row per subagent
#
# Env:
#   CLAUDE_PROJECTS   override the projects root (default ~/.claude/projects)
#
# Requires: jq
set -euo pipefail

command -v jq >/dev/null || { echo "error: jq is required (brew install jq)" >&2; exit 1; }

SID=""
SINCE=""
PER_AGENT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE="${2:-}"; shift 2 ;;
    --per-agent) PER_AGENT=1; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) SID="$1"; shift ;;
  esac
done

# Resolve the project transcript directory from the repo path (Claude encodes
# the absolute path with every '/' replaced by '-').
root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
slug="$(printf '%s' "$root" | sed 's#/#-#g')"
proj="${CLAUDE_PROJECTS:-$HOME/.claude/projects}/$slug"
[ -d "$proj" ] || { echo "error: no transcripts at $proj" >&2; exit 1; }

# Pick the session file.
if [ -n "$SID" ]; then
  f="$proj/$SID.jsonl"
else
  f="$(ls -t "$proj"/*.jsonl 2>/dev/null | head -1 || true)"
  SID="$(basename "${f:-}" .jsonl)"
fi
[ -f "$f" ] || { echo "error: session file not found: $f" >&2; exit 1; }

# jq filter clause for the optional --since lower bound (lexical ISO compare).
since_sel='true'
[ -n "$SINCE" ] && since_sel="((.timestamp // \"0\") >= \"$SINCE\")"

# Portable thousands separator (BSD/GNU awk both fine — no sed GNU extensions).
comma() {
  awk -v n="${1:-0}" 'BEGIN{
    neg=(n<0); if(neg)n=-n; s=sprintf("%d",n); out="";
    while(length(s)>3){ out=","substr(s,length(s)-2)out; s=substr(s,1,length(s)-3) }
    printf "%s%s%s", (neg?"-":""), s, out
  }'
}

size="$(du -h "$f" | cut -f1)"
mtime="$(date -r "$f" '+%Y-%m-%d %H:%M' 2>/dev/null || stat -c '%y' "$f" 2>/dev/null | cut -d. -f1)"

echo "RedLilium — session stats"
echo "  project : $root"
echo "  session : $SID  ($size, $mtime)"
[ -n "$SINCE" ] && echo "  since   : $SINCE"
echo

# --- Main loop, by model ---
echo "Main loop — by model (the tier that owned the session):"
jq -rn --argjson _ 0 "
  [inputs
   | select(.type==\"assistant\" and .message.usage and $since_sel)
   | {m:(.message.model // \"?\"),
      o:(.message.usage.output_tokens // 0),
      i:(.message.usage.input_tokens // 0),
      cc:(.message.usage.cache_creation_input_tokens // 0),
      cr:(.message.usage.cache_read_input_tokens // 0)}]
  | group_by(.m)[]
  | [.[0].m, (length|tostring),
     (map(.o)|add|tostring), (map(.i)|add|tostring),
     (map(.cc)|add|tostring), (map(.cr)|add|tostring)]
  | @tsv
" "$f" 2>/dev/null | while IFS=$'\t' read -r m msgs o i cc cr; do
  [ "$m" = "<synthetic>" ] && continue
  printf "  %-24s msgs=%-6s output=%-13s input=%-11s cache_creation=%-12s cache_read=%s\n" \
    "$m" "$msgs" "$(comma "$o")" "$(comma "$i")" "$(comma "$cc")" "$(comma "$cr")"
done
echo

# --- Subagent launches, by requested type (from the main log's Task calls) ---
echo "Subagent launches — by requested type:"
jq -rn "
  [inputs
   | select(.type==\"assistant\" and $since_sel)
   | .message.content[]?
   | select(.type==\"tool_use\" and (.name==\"Task\" or .name==\"Agent\"))
   | (.input.subagent_type // \"unspecified\")]
  | group_by(.)[] | [length, .[0]] | @tsv
" "$f" 2>/dev/null | sort -rn | while IFS=$'\t' read -r n t; do
  printf "  %-22s ×%s\n" "$t" "$n"
done
echo

# --- Subagents, from their own transcripts (tier + tokens) ---
sub="$proj/$SID/subagents"
if [ -d "$sub" ] && ls "$sub"/agent-*.jsonl >/dev/null 2>&1; then
  rows="$(
    for j in "$sub"/agent-*.jsonl; do
      id="$(basename "$j" .jsonl)"
      meta="$sub/$id.meta.json"
      type="$(jq -r '.agentType // "?"' "$meta" 2>/dev/null || echo '?')"
      jq -rn --arg id "${id#agent-}" --arg type "$type" "
        [inputs | select(.message.usage and $since_sel)
         | {m:(.message.model // \"?\"), o:(.message.usage.output_tokens // 0)}]
        | select(length>0)
        | [.[0].m, \$type, \$id, (length|tostring), (map(.o)|add|tostring)] | @tsv
      " "$j" 2>/dev/null
    done
  )"
  echo "Subagents — by tier:"
  printf '%s\n' "$rows" | awk -F'\t' 'NF{c[$1]++; t[$1]+=$5} END{for(m in c) printf "%s\t%d\t%d\n", m, c[m], t[m]}' \
    | sort -t$'\t' -k3 -rn | while IFS=$'\t' read -r m cnt sum; do
        printf "  %-24s agents=%-4s output=%s\n" "$m" "$cnt" "$(comma "$sum")"
      done
  if [ "$PER_AGENT" = 1 ]; then
    echo
    echo "Subagents — per agent (tier / type / msgs / output):"
    printf '%s\n' "$rows" | sort -t$'\t' -k5 -rn | while IFS=$'\t' read -r m type id msgs o; do
      printf "  %-24s %-16s %-8s msgs=%-4s output=%s\n" "$m" "$type" "${id:0:8}" "$msgs" "$(comma "$o")"
    done
  fi
else
  echo "Subagents — none recorded for this session."
fi
echo
echo "Note: a long-lived/compacted transcript accumulates many turns across"
echo "days and tiers. Scope to one day with --since YYYY-MM-DD. 'output' is the"
echo "cost-relevant axis for tier arbitrage (cache_read is near-free)."
