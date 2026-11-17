We are on branch `domain-agnostic-restructure` of the race engineer project. Read CLAUDE.md and the
memory notes "domain-engineer-roadmap" and "live-replay-testing" first. Do not re-open decisions
recorded there or below.

## Where we are
The playbook step is done: `engineer/playbook.py` is joined into `ROLE_PROMPT`, the LMU energy rule
lives in `domains/lmu/instructions.py`, and the talk rules now say: each lap keeps its own time,
call a lap what the data calls it, never guess a lap type. No data code was changed in that step.

## This step: lap validity, track limits and penalties (data, not prompt)

### What I want
- Lap kinds are: lap, out lap (left the pits), in lap (entered the pits), invalid lap, aborted lap.
- **Invalid lap** means only one thing: the game flagged the lap for track limits and no longer
  counts it. An out lap, an in lap or an aborted lap is never "invalid".
- Practice and qualifying: I care about an invalid lap only because the lap time is gone.
  In and out laps there matter only for measuring the pit time.
- Race: I care about track limits only as penalty risk. Store the track-limit points I have, the
  limit at which a penalty comes, and where each one happened ("exit of turn 2").
- Penalties: how many are outstanding, and what the game says they are.
- Yellow-flag counts are of no interest in LMU: make that something a domain switches on, and LMU
  does not.
- Open question to settle in the design: a jump to the garage in practice or qualifying ends the
  lap as "aborted lap". It must not produce a pit-stop time, because none was driven.

### What the Fuji live test showed (run `data/runs/20261117_142110/`)
1. "So what is going on now?" answered "you've done 1 lap, and it was invalid". The store had that
   lap correctly as "aborted lap", but `session_summary` returned `invalid_laps: 1`, because
   `domains/common/analysis/session/session_summary.py:159` counts every lap the game did not time
   (`sum(not lap.timed ...)`; line 219 does the same for a results file). Out laps, in laps and
   aborted laps are all counted as invalid there.
2. The same answer said "3 yellow flags". `yellow_flags` in the summary (counted in
   `session_log.py`, around line 200) read 1, 3 and 5 within three minutes of a practice session.
3. "What type of lap am I on?" could not be answered: `lap_data` gives `lap_type: "in progress"` for
   the lap being driven (`domains/common/tools/store_tools.py:387`). The type is only known when the
   lap ends (`LapTypeTracker` in `domains/common/analysis/corners/lap_types.py`). An out lap is
   known from the moment the car leaves the pits, so the lap in progress can carry a provisional
   type.
4. `driver_laps` shows `timed: false` for invalid laps and for in, out and aborted laps alike, so
   the model called an untimed lap "probably invalid". The lap type should be on each row.
5. Checked and fine: the "about 30 seconds" pit loss. Laps 8 and 9 took 211.4 s line to line
   against 180.1 s for two laps at the best time, minus about 1 s for a lock-up.

### What the data has (verified in the Fuji recording)
LMU shared memory, read through `thirdparty/pyLMUSharedMemory/lmu_data.py`; none of these are read
by our converters yet, but all are in the recording, so this step can be replay-tested for free:
| Field | Table in the recording | Seen at Fuji |
|---|---|---|
| `mTrackLimitsSteps` (points times steps per point) | player telemetry 100 Hz, cars telemetry 50 Hz | 0 and 1 |
| `mTrackLimitsStepsPerPoint` | scoring info 5 Hz | 4 |
| `mTrackLimitsStepsPerPenalty` | scoring info 5 Hz | 20 |
| `mCountLapFlag` (0 no lap, 1 lap but no time, 2 lap and time) | cars scoring 5 Hz | 1 and 2 |
| `mNumPenalties` (outstanding) | cars scoring 5 Hz | 0 |

The store of that run has laps 4 and 5 as "invalid lap", laps 1, 12 and 19 as "aborted lap".
The results file already gives `<TrackLimits>` and `<Penalty>` events after the session
(`domains/common/data/results.py`); `track_limit_warnings` and `penalties` in the summary stay
empty until then.
The place of a track-limit point can come from the car's position at the moment the steps go up,
using the existing corner segmentation (corner name plus entry, apex or exit).

### How to work
- Design first: folders and files, what each does, why and how. Wait for my OK.
- What any race car has (lap kinds, a track-limit event with its place, penalties) belongs in
  `domains/common/`; reading the LMU fields belongs in `domains/lmu/`.
- Verify with a free replay of the Fuji run first. Tell me the cost before any paid replay.
- Do not run ruff. Run the tests before reporting done.
- Answers to me in chat: short and plain.
