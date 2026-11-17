from ai_race_engineer.domains.common.callouts.base import Callout
from ai_race_engineer.domains.common.callouts.fastest_lap import FastestLap
from ai_race_engineer.domains.common.data.snapshot import Snapshot


class OverallBest(FastestLap):
    """Announces a new overall session best lap (any car, any class) and nothing else."""

    name = "overall_best"

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Callout for a new overall best lap.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "fastest_lap" (facts driver, class_name, lap_s, is_player)
        """
        callout = super().check(snapshot, now_s)
        return callout if callout is not None and callout.kind == "fastest_lap" else None
