// The recording clock, read from the wall.
"use strict";

function pad(n) { return String(n).padStart(2, "0"); }

/* The recording clock, read from the wall. `since` is when it last started running, null while
 * paused; `base` is the seconds banked before that. ⚠ Counting setInterval ticks undercounted:
 * Chromium throttles timers in a hidden or occluded window, and with the display asleep a
 * 51-minute meeting read about 30 while the audio ran the whole time. */
function clockSeconds(clock, now) {
  if (!clock) return 0;
  return Math.floor(clock.base + (clock.since == null ? 0 : (now - clock.since) / 1000));
}

function hhmmss(totalSeconds) {
  const s = Math.floor(totalSeconds);
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

module.exports = { pad, clockSeconds, hhmmss };
