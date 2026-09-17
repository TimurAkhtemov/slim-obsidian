#!/bin/bash
# The launchd job's actual command: export, then sweep. One tick of the hands-off pipeline.
#
# Two steps, in order, because the sweep can only copy files that already exist on disk:
#   1. `shortcuts run` fires the folder-bookmark Shortcut, which exports Voice Memos
#      recordings into the drop folder. Headless execution is CONFIRMED working (2026-07-20);
#      the bookmark's access was granted by the owner's file picker, so no FDA anywhere.
#   2. `slim memos --container` sweeps the drop folder → transcribe → tag → ingest → embed.
#      Idempotent and stem-addressed, so re-exporting the same recordings every tick is
#      harmless: already-swept recordings are skipped, and a memo whose audio has not yet
#      finished syncing is simply caught on a later tick.
#
# A failed export must not abort the sweep — the drop folder may already hold memos from a
# previous tick that still need processing. So step 1 never blocks step 2.
#
# Placeholders (@SHORTCUT@, @CONTAINER@, @UV@) are filled by install-memo-sweep.sh, the same
# way the plist and the backup job are generated — a committed script with real paths would
# be a lie on any other machine.
set -u

SHORTCUT="@SHORTCUT@"
CONTAINER="@CONTAINER@"
UV="@UV@"

# Best-effort export. `|| true` because a Shortcut hiccup is not a reason to skip a sweep
# of what is already on disk.
/usr/bin/shortcuts run "$SHORTCUT" >/dev/null 2>&1 || true

exec "$UV" run slim memos --container "$CONTAINER"
