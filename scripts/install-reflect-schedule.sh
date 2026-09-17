#!/usr/bin/env bash
#
# Install (or re-install) the LaunchAgent that runs scripts/reflect.sh nightly at 03:45.
#
# launchd, not cron: cron is legacy on macOS and skips runs outright when the machine is asleep;
# launchd fires a missed StartCalendarInterval job on the next wake. Same shape as
# install-memo-sweep.sh and install-backup-schedule.sh (absolute path resolution, log location,
# LaunchAgents path).
#
# The plist is GENERATED here rather than committed: it needs absolute paths (repo root, uv's bin
# dir, $HOME), which are machine-specific. A committed plist would be a lie on any other machine.
#
#   ./scripts/install-reflect-schedule.sh          # install + load
#   ./scripts/install-reflect-schedule.sh --remove # unload + delete
#
# NIGHTLY at 03:45, deliberately NOT a short StartInterval: the journals for the day are done
# by then, and a model pass that writes a note should not run while they are working. RunAtLoad is
# FALSE on purpose: installing the agent must NOT fire a live, note-writing reflection (and a
# model pass) the moment it is installed.
#
# To change the cadence (the one knob): edit the StartCalendarInterval Hour/Minute below and re-run
# this script. Keep it clear of the 12:30 backup.
#
# NO Full Disk Access anywhere. reflect reads the brain DB, the vault, and the local model and writes
# a note into the vault — all user-accessible paths, none TCC-protected. Do NOT grant FDA to
# uv/python/bash: FDA on an interpreter is a permanent TCC bypass for every script on the machine
# (standing rule — see CLAUDE.md and the decision log).

set -euo pipefail

LABEL="com.slim.reflect"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/slim-reflect.log"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="gui/$(id -u)"

die() { printf 'install-reflect-schedule: %s\n' "$1" >&2; exit 1; }

if [[ "${1:-}" == "--remove" ]]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed: $LABEL"
  exit 0
fi

command -v uv >/dev/null 2>&1 || die "uv is not installed (see https://docs.astral.sh/uv/)"
[[ -x "$REPO_ROOT/scripts/reflect.sh" ]] || die "scripts/reflect.sh is missing or not executable"

# launchd gives a job a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) that excludes uv's dir
# (~/.local/bin). Pin its ABSOLUTE dir into the agent now — or every scheduled run dies at
# `command -v uv` (and must never fall back to the system python 3.9).
UV_DIR="$(dirname "$(command -v uv)")"

mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$REPO_ROOT/scripts/reflect.sh</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$UV_DIR:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>$REPO_ROOT</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>45</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG</string>
    <key>StandardErrorPath</key>
    <string>$LOG</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST_EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"

cat <<INFO
installed: $LABEL
  runs:  $REPO_ROOT/scripts/reflect.sh   ($(command -v uv) run slim reflect)
  when:  nightly at 03:45 (fires on next wake if the Mac was asleep)
  log:   $LOG

Verify:   launchctl print $DOMAIN/$LABEL | grep -iE 'state|periodic|calendar|program'
Unload:   launchctl bootout $DOMAIN/$LABEL
Reload after editing the time: launchctl bootout $DOMAIN/$LABEL && launchctl bootstrap $DOMAIN "$PLIST"

To change the cadence, edit the StartCalendarInterval Hour/Minute above and re-run this script
(it boots out the old agent and re-bootstraps the new plist).

Deliberately NO kickstart here: a live reflection runs a full model pass and
writes a note. Let the 03:45 timer fire it, or preview by hand with:
  ./scripts/reflect.sh --dry-run
INFO
