from collections.abc import Callable

from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.data.snapshot import Snapshot

PRE_RACE: str = "pre_race"


class PlanBriefing(Event):
    """Reads the race plan to the driver once before the start: the laps, the stops, the lap to box and the energy
    to leave the pits with. The sentence is fixed; nothing is asked of a model.
    _plan    Callable[[], dict[str, object]]  the race plan (plan_basis, race_laps, stops, pit_at_end_of_lap)
    _energy  Callable[[], dict[str, object]]  the pit energy figures (leave_pits_with_percent)
    _hold_s  float                            s the plan must stay the same before it is read
    _said    bool                             the plan of this session was read
    _seen    tuple[object, ...] | None        the plan figures at the last check
    _since_s float                            s monotonic clock since when the plan has those figures
    """

    name = "plan_briefing"

    def __init__(
        self, plan: Callable[[], dict[str, object]], energy: Callable[[], dict[str, object]], hold_s: float
    ) -> None:
        """Store the sources of the plan.
        In:  plan    Callable[[], dict[str, object]]  the race plan tool
             energy  Callable[[], dict[str, object]]  the pit energy tool
             hold_s  float                            s the plan must stay the same before it is read
        Out:         None
        """
        self._plan: Callable[[], dict[str, object]] = plan
        self._energy: Callable[[], dict[str, object]] = energy
        self._hold_s: float = hold_s
        self._said: bool = False
        self._seen: tuple[object, ...] | None = None
        self._since_s: float = 0.0

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Callout with the plan, once, while the plan is still the one made before the start and after it has
        stayed the same for hold_s: in the first seconds of a session the car's energy is not read yet and the plan
        still moves.
        In:  snapshot  Snapshot  (unused: the plan comes from the tools)
             now_s     float     s monotonic clock
        Out:           Callout | None  kind "plan_briefing"; facts race_laps, stops
        """
        if self._said:
            return None
        plan = self._plan()
        if plan.get("plan_basis") != PRE_RACE:
            return None
        figures = (plan["race_laps"], plan["stops"], plan.get("pit_at_end_of_lap"))
        if figures != self._seen:
            self._seen, self._since_s = figures, now_s
        if now_s - self._since_s < self._hold_s:
            return None
        self._said = True
        laps, stops = plan["race_laps"], plan["stops"]
        text = f"{laps} laps. No stop needed."
        if stops:
            leave = self._energy().get("leave_pits_with_percent")
            text = f"{laps} laps. {stops} stop{'s' if stops > 1 else ''}, box end of lap {plan['pit_at_end_of_lap']}"
            text += "." if leave is None else f", add energy to {leave:.0f} percent."
        return Callout("plan_briefing", text, Priority.STRATEGY, {"race_laps": laps, "stops": stops})

    def reset(self) -> None:
        """Allow the plan of a new session to be read.
        In:  -
        Out: None
        """
        self._said = False
        self._seen = None
