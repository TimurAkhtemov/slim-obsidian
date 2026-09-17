#!/usr/bin/env bash
#
# Install (or re-install) the LaunchAgent that runs scripts/backup.sh daily.
#
# launchd, not cron: cron is legacy on macOS and skips runs outright when the machine is
# asleep. launchd fires a missed StartCalendarInterval job on the next wake instead — which
# matters for a laptop that is closed at 12:30 most days.
#
# The plist is GENERATED here rather than committed: it needs absolute paths (repo root,
# restic's bin dir, $HOME), and those are machine-specific. A committed plist would be a lie
# on any other machine.
#
#   ./scripts/install-backup-schedule.sh          # install + load
#   ./scripts/install-backup-schedule.sh --remove # unload + delete
#
# ⚠️  ONE MANUAL STEP THIS SCRIPT CANNOT DO FOR YOU:
#     ~/Documents is TCC-protected. A launchd job cannot read the vault until you grant
#     Full Disk Access to restic ONLY:
#       System Settings -> Privacy & Security -> Full Disk Access -> /opt/homebrew/bin/restic
#
#     Do NOT grant Full Disk Access to /bin/bash. FDA on a general-purpose interpreter is not
#     a narrow grant — it hands a permanent TCC bypass to EVERY shell script on the machine,
#     including anything you download and run. restic is a purpose-built binary whose whole
#     job is reading files to back them up; that is the bounded grant TCC's exemption exists
#     for. backup.sh is written so bash never needs to read the vault itself.
#
#     Without the grant, restic fails "operation not permitted" — IN THE BACKGROUND, where you
#     will not see it. Verify with a real run after granting (see this script's output).

set -euo pipefail

LABEL="com.slim.backup"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/slim-backup.log"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="gui/$(id -u)"

die() { printf 'install-backup-schedule: %s\n' "$1" >&2; exit 1; }

if [[ "${1:-}" == "--remove" ]]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed: $LABEL"
  exit 0
fi

command -v restic >/dev/null 2>&1 || die "restic is not installed. Run: brew install restic"
[[ -x "$REPO_ROOT/scripts/backup.sh" ]] || die "scripts/backup.sh is missing or not executable"
[[ -f "$REPO_ROOT/.env" ]] || die ".env not found — the agent needs RESTIC_REPOSITORY + credentials"

# launchd gives a job a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) that does NOT include
# Homebrew. Resolve restic's directory now and pin it into the agent's environment, or every
# scheduled run dies at `command -v restic`.
RESTIC_DIR="$(dirname "$(command -v restic)")"

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
        <string>$REPO_ROOT/scripts/backup.sh</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$RESTIC_DIR:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>$REPO_ROOT</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>12</integer>
        <key>Minute</key>
        <integer>30</integer>
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
  runs:  $REPO_ROOT/scripts/backup.sh
  when:  daily at 12:30 (fires on next wake if the Mac was asleep)
  log:   $LOG

Next:
  1. Grant Full Disk Access to restic ONLY -- NOT /bin/bash:
       System Settings -> Privacy & Security -> Full Disk Access -> $(command -v restic)
     (FDA on bash would give every shell script on this Mac a permanent TCC bypass.)
  2. Force a real run:   launchctl kickstart -k $DOMAIN/$LABEL
  3. Read the log:       tail -20 "$LOG"
  4. Confirm freshness:  (set -a; source "$REPO_ROOT/.env"; set +a; restic snapshots --latest 1)

A scheduled backup that fails quietly is worse than a manual one you watch succeed.
Step 2-4 is what makes this real rather than hopeful.
INFO
