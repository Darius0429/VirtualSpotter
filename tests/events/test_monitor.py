import queue
from collections.abc import Callable

from ai_race_engineer.events.base import Callout, Event, Priority
from ai_race_engineer.events.monitor import EventMonitor
from ai_race_engineer.strategy.consumption import ConsumptionEstimate, ConsumptionTracker
from ai_race_engineer.telemetry.base import TelemetrySource
from ai_race_engineer.telemetry.snapshot import SessionType, Snapshot, VehicleState


class FakeSource(TelemetrySource):
    """Source returning scripted reads in order.
    _reads  list[Snapshot | None]
    """

    def __init__(self, reads: list[Snapshot | None]) -> None:
        """Store the scripted reads.
        In:  reads  list[Snapshot | None]
        Out:        None
        """
        self._reads: list[Snapshot | None] = list(reads)

    def open(self) -> None:
        """Nothing to open.
        In:  -
        Out: None
        """

    def read(self) -> Snapshot | None:
        """Next scripted read.
        In:  -
        Out: Snapshot | None
        """
        return self._reads.pop(0)

    def close(self) -> None:
        """Nothing to close.
        In:  -
        Out: None
        """


class _Recorder(Event):
    """Event recording checks (with the consumption estimate seen) and resets, returning a fixed callout.
    checks   list[ConsumptionEstimate]
    resets   int
    _tracker  ConsumptionTracker
    _callout  Callout | None
    """

    def __init__(self, tracker: ConsumptionTracker, callout: Callout | None) -> None:
        """Set the tracker to observe and the callout to return.
        In:  tracker  ConsumptionTracker
             callout  Callout | None
        Out:          None
        """
        self.checks: list[ConsumptionEstimate] = []
        self.resets: int = 0
        self._tracker: ConsumptionTracker = tracker
        self._callout: Callout | None = callout

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Record the estimate and return the fixed callout.
        In:  snapshot  Snapshot
             now_s     float     s
        Out:           Callout | None
        """
        self.checks.append(self._tracker.estimate)
        return self._callout

    def reset(self) -> None:
        """Count the reset.
        In:  -
        Out: None
        """
        self.resets += 1


class _Alpha(_Recorder):
    """Recorder named alpha."""

    name = "alpha"


class _Beta(_Recorder):
    """Recorder named beta."""

    name = "beta"


ALPHA = Callout("alpha", "a", Priority.INFO, {})
BETA = Callout("beta", "b", Priority.SPOTTER, {})
ENABLED = {"alpha": ["RACE", "PRACTICE"], "beta": ["RACE"]}


def _monitor(
    reads: list[Snapshot | None],
) -> tuple[EventMonitor, _Alpha, _Beta, ConsumptionTracker, queue.Queue[Callout]]:
    """Monitor over scripted reads with two recording events and a seeded tracker.
    In:  reads  list[Snapshot | None]
    Out:        tuple[EventMonitor, _Alpha, _Beta, ConsumptionTracker, queue.Queue[Callout]]
    """
    tracker = ConsumptionTracker(3, fuel_seed_l_per_lap=3.0)
    alpha, beta = _Alpha(tracker, ALPHA), _Beta(tracker, BETA)
    callouts: queue.Queue[Callout] = queue.Queue()
    return (
        EventMonitor(FakeSource(reads), [alpha, beta], ENABLED, tracker, callouts, 20.0),
        alpha,
        beta,
        tracker,
        callouts,
    )


def _run(monitor: EventMonitor, steps: int) -> None:
    """Step the monitor a number of times.
    In:  monitor  EventMonitor
         steps    int
    Out:          None
    """
    for i in range(steps):
        monitor.step(float(i))


def test_none_read_queues_nothing() -> None:
    """A None read does nothing."""
    monitor, alpha, _, _, callouts = _monitor([None])
    _run(monitor, 1)
    assert callouts.empty()
    assert alpha.checks == []


def test_only_enabled_events_run(make_snapshot: Callable[..., Snapshot]) -> None:
    """Events run only in the session types they are enabled for."""
    monitor, alpha, beta, _, callouts = _monitor([make_snapshot(session_type=SessionType.PRACTICE)])
    _run(monitor, 1)
    assert len(alpha.checks) == 1
    assert beta.checks == []
    assert callouts.get_nowait() == ALPHA
    assert callouts.empty()


def test_callouts_queued_in_order(make_snapshot: Callable[..., Snapshot]) -> None:
    """Callouts are queued in event order."""
    monitor, _, _, _, callouts = _monitor([make_snapshot()])
    _run(monitor, 1)
    assert [callouts.get_nowait(), callouts.get_nowait()] == [ALPHA, BETA]


def test_consumption_updated_before_events(make_snapshot: Callable[..., Snapshot]) -> None:
    """Events see the estimate from this tick's update."""
    monitor, alpha, _, _, _ = _monitor([make_snapshot()])
    _run(monitor, 1)
    assert alpha.checks[0].fuel_l_per_lap == 3.0


def test_first_snapshot_is_new_session(make_snapshot: Callable[..., Snapshot]) -> None:
    """The first snapshot resets events."""
    monitor, alpha, beta, _, _ = _monitor([make_snapshot()])
    _run(monitor, 1)
    assert (alpha.resets, beta.resets) == (1, 1)


def _lap(make_vehicle: Callable[..., VehicleState], laps: int) -> VehicleState:
    """Player vehicle with the given completed laps.
    In:  make_vehicle  Callable[..., VehicleState]
         laps          int
    Out:               VehicleState
    """
    return make_vehicle(slot_id=0, is_player=True, completed_laps=laps)


def _assert_new_session_resets(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot], **changed: object
) -> None:
    """After two measured laps, a snapshot with the changed session values resets events and consumption.
    In:  make_vehicle   Callable[..., VehicleState]
         make_snapshot  Callable[..., Snapshot]
         changed        object  make_snapshot session overrides for the third snapshot
    Out:                None
    """
    tracker = ConsumptionTracker(3)
    alpha = _Alpha(tracker, None)
    reads = [
        make_snapshot(_lap(make_vehicle, 0), fuel_l=50.0, elapsed_s=100.0),
        make_snapshot(_lap(make_vehicle, 1), fuel_l=47.0, elapsed_s=200.0),
        make_snapshot(_lap(make_vehicle, 1), fuel_l=47.0, **{"elapsed_s": 300.0, **changed}),
    ]
    enabled = {"alpha": [t.name for t in SessionType]}
    monitor = EventMonitor(FakeSource(reads), [alpha], enabled, tracker, queue.Queue(), 20.0)
    _run(monitor, 2)
    assert tracker.estimate.fuel_l_per_lap == 3.0
    _run(monitor, 1)
    assert alpha.resets == 2
    assert tracker.estimate.fuel_l_per_lap is None


def test_session_type_change_resets(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """A session type change resets events and consumption."""
    _assert_new_session_resets(make_vehicle, make_snapshot, session_type=SessionType.QUALIFY)


def test_track_change_resets(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """A track change resets events and consumption."""
    _assert_new_session_resets(make_vehicle, make_snapshot, track_name="Other")


def test_elapsed_backwards_resets(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """The session clock going backwards resets events and consumption."""
    _assert_new_session_resets(make_vehicle, make_snapshot, elapsed_s=50.0)


def test_none_between_same_session_no_reset(make_snapshot: Callable[..., Snapshot]) -> None:
    """A None read between snapshots of the same session does not reset."""
    monitor, alpha, _, _, _ = _monitor([make_snapshot(elapsed_s=100.0), None, make_snapshot(elapsed_s=101.0)])
    _run(monitor, 3)
    assert alpha.resets == 1
    assert len(alpha.checks) == 2
