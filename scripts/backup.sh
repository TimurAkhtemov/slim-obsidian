#!/usr/bin/env bash
#
# Back up the one SOURCE-CLASS artifact. Re-runnable; safe to cron.
#
#   the vault   wherever `slim/vaultpath.py` finds it (today, iCloud's Obsidian container)
#     The ONLY copy of every recording and note. iCloud is sync, not backup:
#     a delete or a corrupt write propagates to every device. Risk register: "No backups |
#     High | Certain today."
#
# The brain DB (~/Library/Application Support/slim/) is deliberately NOT backed up: it is
# derived state by design — chunks, embeddings, FTS — reconstructible via `slim ingest`.
# Backing it up would imply it is a source of truth. It is not. (The curation ledger this
# script used to carry as a second artifact was deleted with curated memory, 2026-08-27.)
#
# Setup (once):
#   brew install restic
#   Add to .env (gitignored):
#     export RESTIC_REPOSITORY="b2:your-bucket:slim"      # or s3:..., sftp:..., /Volumes/...
#     export RESTIC_PASSWORD_FILE="$HOME/.config/slim/restic-password"
#     export B2_ACCOUNT_ID="..."                          # if using B2
#     export B2_ACCOUNT_KEY="..."
#   restic init          # once, after the env is set
#
# Usage:
#   ./scripts/backup.sh              # backup + prune + fast metadata check
#   ./scripts/backup.sh --verify     # also re-read 5% of pack data (slower, catches bitrot)

set -euo pipefail

# Any unexpected failure MUST say so. Without this, `set -e` + `pipefail` + a suppressed
# stderr exits 1 with a completely empty log — which under launchd is indistinguishable from
# "nothing happened", and a backup that fails silently is worse than no backup at all.
# (This is not hypothetical: the iCloud probe below did exactly that on its first scheduled run.)
trap 's=$?; printf "backup: FAILED (exit %s) at line %s: %s\n" "$s" "$LINENO" "$BASH_COMMAND" >&2; exit $s' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .env is gitignored and holds the repo, credentials and any persistent vault override. Load it
# before discovery: launchd supplies only PATH, so an SLIM_VAULT set here is otherwise invisible.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi

# ⚠ ASK, do not assume. This ran for months against a hardcoded path; the day the vault moved
# into iCloud for the iOS app, `die` below would have failed the launchd job every night and
# said so only in a log nobody reads. `vaultpath.py` is stdlib-only for exactly this caller —
# the system python, no venv.
VAULT="${SLIM_VAULT:-$(/usr/bin/python3 "$REPO_ROOT/slim/vaultpath.py" 2>/dev/null)}"

die() { printf 'backup: %s\n' "$1" >&2; exit 1; }

# --- preconditions -------------------------------------------------------------------------
command -v restic >/dev/null 2>&1 || die "restic is not installed. Run: brew install restic"

[[ -n "${RESTIC_REPOSITORY:-}" ]] || die "RESTIC_REPOSITORY is unset (see the setup block above)"
[[ -n "${RESTIC_PASSWORD_FILE:-}${RESTIC_PASSWORD:-}" ]] || die "RESTIC_PASSWORD_FILE or RESTIC_PASSWORD is unset"

[[ -d "$VAULT" ]]  || die "vault not found: $VAULT"

# ~/Documents is TCC-protected. ONLY restic is granted Full Disk Access — deliberately NOT
# /bin/bash, because FDA on a general-purpose interpreter hands a TCC bypass to every shell
# script on the machine. The cost of that choice is here: under launchd, bash cannot read the
# vault, so the iCloud placeholder check below is skipped and restic is the sole reader.
# It is NOT fatal — restic still has access and will fail loudly (ERR trap) if it does not,
# and the file-count check after the backup covers the silent case.
if ls "$VAULT" >/dev/null 2>&1; then
  # The vault lives in iCloud. An evicted file becomes a 0-byte '.icloud' placeholder stub, and
  # a backup of stubs is a backup of nothing — the exact silent failure this script exists to
  # prevent. Fail loudly; `brctl download "$VAULT"` materializes them.
  placeholders="$(find "$VAULT" -name '*.icloud' 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$placeholders" == "0" ]] || die "$placeholders iCloud placeholder(s) in the vault — files are evicted, not local.
       Run: brctl download \"$VAULT\"   then re-run this backup."
else
  printf 'backup: NOTE — no shell read access to the vault (TCC); skipping the iCloud\n'
  printf '        placeholder check. restic reads it directly; its file count is checked below.\n'
fi

restic cat config >/dev/null 2>&1 || die "repository unreachable. Check .env (B2 credentials, RESTIC_REPOSITORY) and the network first. Only a BRAND-NEW repository needs 'restic init' — running it against a mis-pointed bucket creates an empty repo and every later backup lands there."

# --- backup --------------------------------------------------------------------------------
printf 'backup: vault=%s\n' "$VAULT"

restic backup \
  --tag slim --tag vault \
  --exclude '.DS_Store' \
  --exclude '*.icloud' \
  --exclude '.Trash' \
  "$VAULT"

# --- did restic read the whole vault? ------------------------------------------------------
# Under launchd the placeholder check above never runs, so compare this snapshot's file count
# with the previous one's. Losing more than a tenth of the files in a day is eviction or a mass
# delete — a clean-looking backup of neither. Stop before `forget` prunes anything.
counts="$(restic snapshots --tag slim --latest 2 --json | /usr/bin/python3 -c '
import json, sys
snaps = sorted(json.load(sys.stdin), key=lambda s: s["time"])
print(" ".join(str((s.get("summary") or {}).get("total_files_processed", "")) for s in snaps))')"
read -r prev now _ <<<"$counts"
if [[ "$prev" =~ ^[0-9]+$ && "$now" =~ ^[0-9]+$ ]]; then
  (( now * 10 >= prev * 9 )) || die "this snapshot read $now files; the previous one read $prev.
       Evicted iCloud files or a mass delete. Nothing was pruned. Check the vault, then re-run."
else
  printf 'backup: NOTE — no previous file count to compare against (%s); skipping the check.\n' "$counts"
fi

# --- retention -----------------------------------------------------------------------------
# Generous on purpose: the archive is irreplaceable and tiny (~400 files). Storage is cheaper
# than a regretted prune.
restic forget \
  --tag slim \
  --keep-daily 14 --keep-weekly 8 --keep-monthly 24 --keep-yearly 10 \
  --prune

# --- verify --------------------------------------------------------------------------------
# An unverified backup is a hope, not a backup.
if [[ "${1:-}" == "--verify" ]]; then
  restic check --read-data-subset=5%
else
  restic check
fi

printf '\nbackup: latest snapshot\n'
restic snapshots --tag slim --latest 1
