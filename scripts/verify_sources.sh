#!/usr/bin/env bash
#
# verify_sources.sh — reproducible citation ledger for CVIAF research.
#
# Reads a list of URLs (one per line, '#' comments allowed) and emits, for each:
#   HTTP status, final URL after redirects, content length, SHA-256 of the body,
#   and the page <title> when one can be extracted.
#
# Why this exists: a citation is only as good as the ability to re-check it.
# Months later a page may have moved or changed — a changed hash is INFORMATION
# (the source moved; re-read any claim resting on it), not a script failure.
#
# Usage:
#   scripts/verify_sources.sh docs/source_urls.txt
#   scripts/verify_sources.sh docs/source_urls.txt > docs/evidence/source_ledger.txt
#
# Offline note: this script needs network access purely to CHECK citations.
# It is a development tool and is NOT part of the CVIAF runtime, which must
# never touch the network. Keep it out of the air-gapped deployment bundle.

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <url-list-file>" >&2
  echo "  one URL per line; lines starting with # are ignored" >&2
  exit 2
fi

URL_FILE="$1"
if [[ ! -f "$URL_FILE" ]]; then
  echo "error: no such file: $URL_FILE" >&2
  exit 2
fi

UA="Mozilla/5.0 (compatible; CVIAF-source-verify/1.0; citation-check)"
TODAY="$(date +%F)"
TIMEOUT="${CURL_TIMEOUT:-30}"

echo "# CVIAF source verification ledger"
echo "# source list : $URL_FILE"
echo "# generated   : $TODAY"
echo "# tool        : scripts/verify_sources.sh"
echo "# curl timeout: ${TIMEOUT}s per URL (override with CURL_TIMEOUT=5)"
echo "#"
echo "# status | final_url | bytes | sha256(first 16) | title"
echo "# ---------------------------------------------------------------"

total=0
failed=0

while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="$(printf '%s' "$raw" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [[ -z "$line" || "$line" == \#* ]] && continue

  total=$((total + 1))
  body="$(mktemp)"
  hdrs="$(mktemp)"

  # -L follow redirects, -sS quiet-but-show-errors, --max-time bounded.
  code="$(curl -L -sS --max-time "$TIMEOUT" -A "$UA" \
            -D "$hdrs" -o "$body" -w '%{http_code}' "$line" 2>/dev/null || echo "000")"

  final="$(grep -i '^location:' "$hdrs" 2>/dev/null | tail -n1 | tr -d '\r' | cut -d' ' -f2-)"
  [[ -z "$final" ]] && final="$line"

  bytes="$(wc -c < "$body" | tr -d ' ')"
  if [[ -s "$body" ]]; then
    hash="$(sha256sum "$body" | cut -c1-16)"
  else
    hash="-"
  fi

  # <title> extraction is best-effort: many valid pages have none.
  title="$(tr '\n' ' ' < "$body" 2>/dev/null \
            | grep -o -i -m1 '<title[^>]*>[^<]*</title>' \
            | sed 's/<[^>]*>//g' | tr -s ' ' | cut -c1-90)"
  [[ -z "$title" ]] && title="(no title)"

  if [[ "$code" != "200" ]]; then
    failed=$((failed + 1))
    title="$title   [NON-200 — treat as UNVERIFIED or GATED]"
  fi

  printf '%s | %s | %s | %s | %s\n' "$code" "$final" "$bytes" "$hash" "$title"

  printf '  requested: %s\n' "$line"

  rm -f "$body" "$hdrs"
done < "$URL_FILE"

echo "# ---------------------------------------------------------------"
echo "# total: $total   non-200: $failed"
echo "#"
echo "# Reminder: a NON-200 is a finding, not a tool error."
echo "#   openaccess.thecvf.com may 403 bot-blocked clients; a browser User-Agent"
echo "#       (which this script sets) usually gets through. If it still fails, cite"
echo "#       the arXiv abs/html page as the fetchable primary and name the venue separately."
echo "#   A changed SHA-256 versus an earlier ledger means the source was edited;"
echo "#       re-read every claim that rested on it."
