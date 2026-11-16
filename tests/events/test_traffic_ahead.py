from types import SimpleNamespace

from ai_race_engineer.events.traffic_ahead import TrafficAhead
from ai_race_engineer.strategy.race_finish import CarInput

TRACK_M: float = 5000.0


def _car(slot_id: int, progress_laps: float, lap_time_s: float, class_name: str = "GT3") -> CarInput:
    """Car at a race distance with a pace and no resources.
    In:  slot_id        int
         progress_laps  float  laps
         lap_time_s     float  s/lap
         class_name     str
    Out:                CarInput
    """
    return CarInput(slot_id, class_name, progress_laps, lap_time_s, False, False, None, None, None, None, None, None)


class _Tracker:
    """Stand-in for the race tracker: the player (slot 0, Hyper, 100 s laps) at a race distance among other cars.
    state     SimpleNamespace | None
    finished  bool
    """

    def __init__(self) -> None:
        """Start without a state.
        In:  -
        Out: None
        """
        self.state: SimpleNamespace | None = None
        self.finished: bool = False

    def at(
        self,
        time_s: float,
        progress_laps: float,
        others: list[CarInput],
        before_green: bool = False,
        parked: tuple[int, ...] = (),
    ) -> None:
        """Set the race state.
        In:  time_s         float           s session clock
             progress_laps  float           laps of the player
             others         list[CarInput]
             before_green   bool
             parked         tuple[int, ...]  slot ids of cars standing in their garage
        Out:                None
        """
        self.state = SimpleNamespace(
            time_s=time_s,
            player_slot_id=0,
            cars=(_car(0, progress_laps, 100.0, "Hyper"), *others),
            pit=SimpleNamespace(track_length_m=TRACK_M),
            vehicles=[SimpleNamespace(slot_id=car.slot_id, in_garage=car.slot_id in parked) for car in others],
            before_green=before_green,
        )


def _texts(tracker: _Tracker, event: TrafficAhead, steps: list[tuple[float, list[CarInput]]]) -> list[str | None]:
    """Sentences the event gives over a run of race states.
    In:  tracker  _Tracker
         event    TrafficAhead
         steps    list[tuple[float, list[CarInput]]]  the player's laps and the other cars at each state
    Out:          list[str | None]                    None = nothing said
    """
    texts: list[str | None] = []
    for time_s, (progress_laps, others) in enumerate(steps):
        tracker.at(float(time_s), progress_laps, others)
        callout = event.check(None, 0.0)
        texts.append(None if callout is None else callout.text)
    return texts


def test_pack_single_car_and_backmarkers_have_their_own_sentence() -> None:
    """Several cars of another class close together, one car, and lapped cars of the own class."""
    pack = [_car(5, 18.55, 110.0), _car(6, 18.56, 110.0)]
    for others, text in (
        (pack, "GT traffic in the next lap"),
        (pack[:1], "GT3 car up ahead"),
        ([_car(5, 18.52, 104.0, "Hyper"), _car(6, 18.53, 104.0, "Hyper")], "Backmarkers in the next lap"),
        ([_car(5, 18.52, 104.0, "Hyper")], "Backmarker up ahead"),
        ([_car(5, 18.55, 110.0, "LMP2"), _car(6, 18.56, 110.0, "LMP2")], "LMP2 traffic in the next lap"),
    ):
        tracker = _Tracker()
        assert _texts(tracker, TrafficAhead(tracker, 1.0, 100.0), [(20.5, others)]) == [text]


def test_traffic_is_said_once_until_it_has_been_passed() -> None:
    """Nothing while the pack is more than a lap away, once when it comes within a lap, not again while it is near,
    and again when the same cars come up a lap later."""
    far = [_car(5, 18.70, 110.0), _car(6, 18.71, 110.0)]
    near = [_car(5, 18.55, 110.0), _car(6, 18.56, 110.0)]
    passed = [_car(5, 18.45, 110.0), _car(6, 18.46, 110.0)]
    tracker = _Tracker()
    event = TrafficAhead(tracker, 1.0, 100.0)
    texts = _texts(tracker, event, [(20.5, far), (20.5, near), (20.52, near), (20.5, passed), (20.5, near)])
    assert texts == [None, "GT traffic in the next lap", None, None, "GT traffic in the next lap"]


def test_nothing_is_said_before_the_green_or_without_a_new_state() -> None:
    """Before the start and on a state already looked at the event is silent."""
    near = [_car(5, 18.55, 110.0), _car(6, 18.56, 110.0)]
    tracker = _Tracker()
    event = TrafficAhead(tracker, 1.0, 100.0)
    tracker.at(1.0, 20.5, near, before_green=True)
    assert event.check(None, 0.0) is None
    tracker.at(2.0, 20.5, near)
    assert event.check(None, 0.0).text == "GT traffic in the next lap"
    assert event.check(None, 0.0) is None


def test_cars_joining_traffic_already_announced_are_not_said_again() -> None:
    """A third car that comes within reach while the pack is still ahead adds no sentence; backmarkers are a
    kind of their own and are said."""
    pack = [_car(5, 18.55, 110.0), _car(6, 18.56, 110.0)]
    more = [*pack, _car(7, 18.60, 110.0)]
    lapped = [*more, _car(8, 18.52, 104.0, "Hyper")]
    tracker = _Tracker()
    event = TrafficAhead(tracker, 1.0, 100.0)
    texts = _texts(tracker, event, [(20.5, pack), (20.5, more), (20.5, lapped), (20.5, lapped)])
    assert texts == ["GT traffic in the next lap", None, "Backmarker up ahead", None]


def test_car_standing_in_its_garage_is_no_traffic() -> None:
    """A car parked in its garage is never announced."""
    tracker = _Tracker()
    event = TrafficAhead(tracker, 1.0, 100.0)
    tracker.at(1.0, 20.5, [_car(5, 18.52, 104.0, "Hyper")], parked=(5,))
    assert event.check(None, 0.0) is None
