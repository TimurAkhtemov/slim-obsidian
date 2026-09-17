#!/bin/bash
# Warn when the `slim chat` on :7546 is running code older than what is on disk.
#
# CLAUDE.md, twice measured: a full day of live testing hit a server started the previous
# evening (2026-07-21), and two recordings failed against an already-fixed bug (2026-08-23).
# `GET /api/health` was built for exactly this; nobody remembers to call it. This does.
#
# A 404 is itself proof of staleness — a server too old to have the endpoint. So anything
# other than a healthy `"stale": false` is reported, and nothing here ever fails the session.

PORT=${SLIM_CHAT_PORT:-7546}   # overridable so the hook itself is testable
BODY=$(curl -sf --max-time 2 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null)

if [[ -z "$BODY" ]]; then
  if lsof -tiTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠ Something is listening on :${PORT} but /api/health did not answer — that is a server"
    echo "  too old to have the endpoint. Kill it before trusting any live behavior:"
    echo "    lsof -tiTCP:${PORT} -sTCP:LISTEN | xargs kill"
  fi
  exit 0   # nothing on the port is the normal case, not a problem
fi

if echo "$BODY" | grep -q '"stale": *true'; then
  NEWEST=$(echo "$BODY" | sed -n 's/.*"newest_file": *"\([^"]*\)".*/\1/p')
  STARTED=$(echo "$BODY" | sed -n 's/.*"started_at": *"\([^"]*\)".*/\1/p')
  echo "⚠ STALE SERVER on :${PORT} — it started ${STARTED} and ${NEWEST} has been saved since."
  echo "  Live behavior does NOT reflect the code on disk. Before debugging anything live:"
  echo "    lsof -tiTCP:${PORT} -sTCP:LISTEN | xargs kill"
  echo "  While developing, prefer: uv run slim chat --reload"
fi
exit 0
