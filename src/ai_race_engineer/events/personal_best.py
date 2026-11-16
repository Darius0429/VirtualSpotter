from ai_race_engineer.events.base import Callout, Priority
from ai_race_engineer.events.fastest_lap import FastestLap
from ai_race_engineer.events.text import lap_time_text
from ai_race_engineer.telemetry.snapshot import Snapshot


class PersonalBest(FastestLap):
    """Announces the player's new personal best lap and nothing else, also when it is the overall best."""

    name = "personal_best"

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Callout for a new personal best lap of the player.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "personal_best" (facts lap_s)
        """
        callout = super().check(snapshot, now_s)
        if callout is None or callout.kind == "personal_best":
            return callout
        if not callout.facts["is_player"]:
            return None
        lap_s = float(callout.facts["lap_s"])
        return Callout(
            "personal_best", f"You set your fastest lap with a {lap_time_text(lap_s)}", Priority.INFO, {"lap_s": lap_s}
        )
