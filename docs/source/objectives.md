# Engineer objectives

What the race engineer must be able to tell the driver, in qualifying and in the race. Designs for `strategy/`,
`events/` and the agent tools are checked against this list.

Status: draft. Items marked **(proposed)** are defaults that the project owner has not confirmed yet.

## How to read this

Each objective has:

- **Driver gets**: the answer or callout, in the driver's words.
- **Mode**: `ask` (answered when the driver asks), `callout` (spoken unprompted), or both. **(proposed)**
- **Data**: where it comes from, and whether a recording has shown that the game publishes it.

Data sources:

| Source | Rate | Content |
|---|---|---|
| Player telemetry (shared memory) | 100 Hz | Inputs, motion, fuel, energy, tyres, suspension, in-car settings |
| Other cars' telemetry (shared memory) | 50 Hz | Position, motion, fuel, virtual energy, tyre compound and wear |
| Scoring (shared memory) | 5 Hz | Places, lap and sector times, gaps, pit state, flags, session phase, weather |
| Game REST API (`localhost:6397`) | on change | Garage setup, pit-stop estimate, brake wear, damage, forecast, session settings |

Known limits:

- Other cars do not publish tyre temperatures or pressures, brake data, hybrid state or their setup.
- A crash is not published as an event. It has to be inferred from a car that is stopped or very slow on track.
- The game's pit-stop estimate was 4.4 s for a stop that took 6.8 s stationary (one stop observed). Treat it as
  service time until more stops are recorded.

## Definitions

### Rival **(proposed)**

Rivals are always in the driver's own class. There are three groups, and a car can be in more than one:

| Group | Who | Changes when |
|---|---|---|
| Direct | The car ahead and the car behind in class | A position changes |
| Pace | Cars whose recent lap times are within a few tenths of the driver's | Lap times drift |
| Strategy | Cars that are far away on track but will be close once all cars have pitted | Someone pits, or fuel and energy plans change |

In qualifying only the first two apply: the cars in the positions around the driver, and the cars with a similar
best lap.

### Limiting resource

In LMU the stint length is set by fuel or by virtual energy, whichever runs out first. Every "laps left" and
"when do I pit" answer uses the limiting one and says which it is.

## 1. Rivals

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 1.1 | "Who are my rivals?" The current direct, pace and strategy rivals | ask | Scoring: place, class, lap times. Confirmed |
| 1.2 | A callout when a rival changes: a new car behind, a rival pitted, a strategy rival is now a direct one | callout | Scoring: place, pit state. Confirmed |
| 1.3 | Qualifying: who is just ahead and behind on the time sheet, and by how much | ask | Scoring: best lap per car. Confirmed |

## 2. Me against rivals

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 2.1 | Lap time against a rival: last lap, best lap, average over the last few laps | ask | Scoring. Confirmed |
| 2.2 | Sector times against a rival, and which sector the gap comes from | ask | Scoring: sector times for all cars. Confirmed |
| 2.3 | Trend: gaining or losing per lap, and when the driver catches the car ahead or is caught | both | Scoring: gaps over laps. Confirmed |
| 2.4 | Fuel and energy against a rival: usage per lap and the lap each is expected to stop | ask | Other cars' telemetry: fuel, virtual energy. Confirmed |
| 2.5 | Tyres against a rival: compound, wear, and degradation read from the lap-time trend | ask | Compound and wear confirmed. No temperatures for rivals |

## 3. Strategy

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 3.1 | Laps left on the limiting resource, and the lap to pit | both | Player telemetry: fuel, energy. Confirmed |
| 3.2 | Fuel-saving advice: whether saving reaches a whole extra lap, how much per lap is needed, whether that is realistic and what it costs in lap time | both | Needs the driver's own saving laps to learn what is realistic. Not recorded yet |
| 3.3 | Laps remaining in a timed race, including whether the leader crosses the line before the clock runs out | both | Scoring: session end time, leader's pace. Confirmed |
| 3.4 | Pit-stop length for the planned service: fuel, energy, tyres, damage, penalties | ask | REST pit-stop estimate. One fuel-only stop recorded |
| 3.5 | Pit-exit prediction: the place and the traffic the driver rejoins in | ask | Pit-lane time measured from recordings (36.8 s at Algarve) plus scoring gaps |
| 3.6 | Undercut warning: a rival has pitted or is about to run out | callout | Scoring pit state, rivals' fuel and energy. Confirmed |

"Realistic" in 3.2 is learned, not assumed: the engineer compares the driver's normal usage per lap with the
lowest usage the driver has shown at a known lap-time cost.

## 4. Track status

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 4.1 | Yellow flag and the sector it is in; green again | callout | Scoring: sector flags. Published, not yet seen raised in a recording |
| 4.2 | Safety car or full-course yellow, and what it does to the pit plan | callout | Scoring: game phase, yellow-flag state. Not yet seen in a recording |
| 4.3 | Blue flag, and a faster class about to lap the driver, with where on the lap | callout | Scoring: per-car flag, class, track position. Published |
| 4.4 | Incident ahead: a stopped or slow car, with corner or distance | callout | Inferred from other cars' speed and track position |
| 4.5 | Slower traffic the driver is about to reach | callout | Scoring: class, gaps. Confirmed |

## 5. Car health and rules

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 5.1 | Damage, and what a repair costs in pit time | both | Player telemetry dents and impacts; REST damage and repair time |
| 5.2 | Brake wear and tyre wear, temperatures and pressures for the driver's own car | ask | Player telemetry and REST. Confirmed |
| 5.3 | Track-limit warnings left, and penalties to serve | both | Player telemetry and scoring. Published, no penalty recorded yet |
| 5.4 | Weather: rain chance, track temperature, and when it changes | both | Scoring and REST forecast. Only dry sessions recorded |

## 6. Qualifying

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 6.1 | Gap to the next position and to pole, and the sector it is in | both | Scoring. Confirmed |
| 6.2 | Theoretical best lap from the driver's best sectors | ask | Scoring. Confirmed |
| 6.3 | Time left: how many flying laps still fit | both | Scoring: session end time. Confirmed |
| 6.4 | Clean-air gap on the out-lap | ask | Other cars' track position. Confirmed |

## 7. Start and formation lap

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 7.1 | Who starts on which compound, with how much fuel and energy | ask | Other cars' telemetry. Confirmed during the formation lap |
| 7.2 | The cars around the driver on the grid and their qualifying pace | ask | Scoring. Confirmed |

## 8. Coaching (later)

| # | Driver gets | Mode | Data |
|---|---|---|---|
| 8.1 | Where time is lost against the driver's own best lap, corner by corner | ask | Player telemetry at 100 Hz. Confirmed |
| 8.2 | Setup hints from ride heights, suspension travel and tyre temperatures across a stint | ask | Player telemetry and REST garage setup. Confirmed |

## Priority **(proposed)**

1. Rivals and gaps (1.x, 2.1 to 2.3): everything else refers to them.
2. Fuel and energy strategy (3.1 to 3.3).
3. Track status callouts (4.x).
4. Pit stop and pit exit (3.4 to 3.6).
5. Qualifying (6.x) and start (7.x).
6. Car health (5.x), then coaching (8.x).

## Open decisions

1. Is the three-group rival definition right, and how wide is "a few tenths" for pace rivals?
2. Is the priority order above right?
3. Is the `ask` / `callout` choice per objective right?
4. Which recordings are still needed: fuel-saving laps, a tyre-change stop, a wet session, a yellow flag or safety
   car, and a penalty.
