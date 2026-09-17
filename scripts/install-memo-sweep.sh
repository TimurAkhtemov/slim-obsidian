#!/usr/bin/env bash
#
# Install (or re-install) the LaunchAgent that runs `slim memos` every 5 minutes: sweep the
# synced Voice Memos container into Inbox/Journal, transcribe, and index (P2 / R4).
#
# ⚠️  THIS SCRIPT PREPARES; IT DOES NOT INSTALL. It generates the plist from
#     com.slim.voicememos.plist.template and prints the exact `launchctl bootstrap` line, but
#     it does NOT run launchctl and does NOT load the agent. Loading the timer is the owner's
#     action — the ACT of scheduling stays theirs. Run the printed
#     command yourself when you are ready.
#
# launchd, not cron: cron is legacy on macOS and skips runs while the machine is asleep;
# launchd fires a missed StartInterval job on the next wake. This mirrors
# install-backup-schedule.sh exactly (absolute path resolution, log location, LaunchAgents
# install path) — read that script alongside this one.
#
#   ./scripts/install-memo-sweep.sh          # generate plist + print the load command
#   ./scripts/install-memo-sweep.sh --remove # print the unload command + delete the plist
#
# THE ACCESS ROUTE (resolved 2026-07-20): NO Full Disk Access, anywhere.
#     macOS Voice Memos has no Shortcuts export action, so a folder-bookmark Shortcut the owner
#     built ("Sync Voice Memos to SLIM") exports recordings into a drop folder — the file
#     picker granted its access by USER CONSENT, which is not TCC and needs no FDA. The job
#     runs that Shortcut (headless `shortcuts run`, confirmed working) and THEN sweeps the
#     drop folder. Nothing here ever reads Apple's protected container directly, so the
#     standing rule (FDA never to an interpreter) is honored by never needing FDA at all.

set -euo pipefail

LABEL="com.slim.voicememos"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
# The name of the macOS Shortcut that exports Voice Memos into the drop folder.
SHORTCUT="${SHORTCUT:-Sync Voice Memos to SLIM}"
# The job's command is a generated wrapper (export then sweep). Generated, not committed —
# it carries this machine's absolute paths, like the plist.
RUNNER="$HOME/Library/Application Support/slim/memo-sweep-run.sh"
LOG="$HOME/Library/Logs/slim-voicememos.log"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_ROOT/scripts/com.slim.voicememos.plist.template"
DOMAIN="gui/$(id -u)"

die() { printf 'install-memo-sweep: %s\n' "$1" >&2; exit 1; }

if [[ "${1:-}" == "--remove" ]]; then
  rm -f "$PLIST"
  echo "removed plist: $PLIST"
  echo "now unload the agent yourself:  launchctl bootout $DOMAIN/$LABEL"
  exit 0
fi

# The folder the macOS Shortcut EXPORTS INTO. Not Apple's Voice Memos container: that stays
# TCC-protected and unreadable to any interpreter, which is precisely why the Shortcut exists
# (its file picker grants access by user consent — no Full Disk Access anywhere).
# Override with: CONTAINER=/some/path ./scripts/install-memo-sweep.sh
CONTAINER="${CONTAINER:-$HOME/Library/Application Support/slim/voicememo-drop}"

[[ -f "$TEMPLATE" ]] || die "missing template: $TEMPLATE"
command -v uv >/dev/null 2>&1 || die "uv is not installed (see https://docs.astral.sh/uv/)"
# ffprobe/ffmpeg are what transcription (slim/transcribe.py, slim/inbox.py) shells out to, and
# they live in Homebrew too — the ORIGINAL install pinned uv's dir but not this one, so the
# launchd sweep failed every tick on `FileNotFoundError: 'ffprobe'` the moment memos landed
# (worked by hand only because an interactive shell has Homebrew on PATH). Pin its dir as well.
command -v ffprobe >/dev/null 2>&1 || die "ffprobe not found (brew install ffmpeg) — transcription needs it"

# launchd hands a job a minimal PATH that excludes Homebrew, so pin uv's absolute path and its
# directory into the agent — exactly as the backup installer pins restic's — AND ffmpeg's dir.
UV="$(command -v uv)"
UV_DIR="$(dirname "$UV")"
FFMPEG_DIR="$(dirname "$(command -v ffprobe)")"

mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")" "$(dirname "$RUNNER")"

# The runner: fire the Shortcut, then sweep the drop folder.
sed -e "s|@SHORTCUT@|$SHORTCUT|g" \
    -e "s|@CONTAINER@|$CONTAINER|g" \
    -e "s|@UV@|$UV|g" \
    "$REPO_ROOT/scripts/memo-sweep-run.sh" > "$RUNNER"
chmod +x "$RUNNER"

sed -e "s|@LABEL@|$LABEL|g" \
    -e "s|@UV@|$UV|g" \
    -e "s|@UV_DIR@|$UV_DIR|g" \
    -e "s|@FFMPEG_DIR@|$FFMPEG_DIR|g" \
    -e "s|@REPO_ROOT@|$REPO_ROOT|g" \
    -e "s|@LOG@|$LOG|g" \
    -e "s|@CONTAINER@|$CONTAINER|g" \
    -e "s|@RUNNER@|$RUNNER|g" \
    "$TEMPLATE" > "$PLIST"

cat <<INFO
prepared (NOT loaded): $PLIST
  runs:  $RUNNER  (shortcuts run "$SHORTCUT"  →  slim memos --container ...)
  when:  every 5 minutes + at login (fires on next wake if the Mac was asleep)
  log:   $LOG

⚠ BEFORE loading this, confirm the Shortcut's "Save File" action actually points at
  $CONTAINER
  and has "Overwrite If File Exists" turned ON. Left at its default, exports land in
  Shortcuts' own iCloud folder while the Shortcut still reports success — the sweep then
  finds nothing and the timer looks broken. Verify with:
      shortcuts run "<your shortcut name>" && ls "$CONTAINER"

This script did NOT load the agent. When you are ready, load it yourself:
  1. Smoke test:  uv run slim memos --dry-run --container "$CONTAINER"
     (The old step here said to validate CloudRecordings.db first. That is OBSOLETE: the
     Shortcut route never touches Apple's container, so its title database is out of reach
     — which is also why exported filenames are timestamps and titles come from enrichment.)
  2. Load the timer:      launchctl bootstrap $DOMAIN "$PLIST"
  3. Kick one run now:    launchctl kickstart -k $DOMAIN/$LABEL
  4. Read the log:        tail -20 "$LOG"

If step 3/4 shows "Operation not permitted", the container is TCC-protected and the sweep
refused (it copied nothing — the safe failure). Do NOT grant Full Disk Access to python/uv or
bash; route the read through a dedicated copy-only helper instead (see this script's header
and the P2 spike). To unload later:  launchctl bootout $DOMAIN/$LABEL
INFO
