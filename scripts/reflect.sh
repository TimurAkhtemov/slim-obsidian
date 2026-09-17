#!/usr/bin/env bash
#
# The nightly `slim reflect` run, invoked by launchd (com.slim.reflect) at 03:45.
#
# WHAT IT DOES: reads the journal entries since the last reflection with the resident model
# and writes ONE reflection note into the vault's `_Reflections/` folder. A quiet window (no new
# journals) writes nothing. Output is a plain markdown note the owner opens in Obsidian — this creates a
# file, it destroys nothing, and `_Reflections/` is excluded from the index so the brain never
# retrieves its own reflections as evidence.
#
# WHY NIGHTLY: the journals for the day are finished by 03:45, and the hour keeps clear of the
# 12:30 backup. Since 2026-09-01 reflect runs on the RESIDENT model (gemma dropped), so nothing
# is evicted any more; the hour stays because it works.
#
# NO Full Disk Access anywhere: reflect reads the brain DB (~/Library/Application Support/slim/),
# the vault, and the local model, and writes a note into the vault — all user-accessible paths, none
# TCC-protected. FDA on an interpreter (uv/python/bash) is a standing NEVER: it hands a permanent
# TCC bypass to every script on the machine.
#
# The plist that runs this is GENERATED + loaded by scripts/install-reflect-schedule.sh. Run this
# by hand any time, and pass flags through:
#   ./scripts/reflect.sh --dry-run        # preview the window; no model call, no write
#   ./scripts/reflect.sh --days 3         # reflect on the last 3 days
set -euo pipefail

# A launchd job that dies silently is indistinguishable from "nothing happened". Say so loudly.
trap 's=$?; printf "reflect: FAILED (exit %s) at line %s: %s\n" "$s" "$LINENO" "$BASH_COMMAND" >&2; exit $s' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# launchd hands the job a minimal PATH; the plist pins uv's dir onto it. Resolve uv to its ABSOLUTE
# path so a stray PATH can never substitute a different interpreter, and so a missing uv fails
# LOUDLY here rather than silently falling back to the system python 3.9.
UV="$(command -v uv)" || { printf 'reflect: uv not found on PATH (%s)\n' "$PATH" >&2; exit 1; }

start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
printf 'reflect: %s starting — %s run slim reflect %s\n' "$start_ts" "$UV" "$*"
"$UV" run slim reflect "$@"
printf 'reflect: %s done (exit 0)\n' "$(date '+%Y-%m-%d %H:%M:%S')"
