from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.callouts.text import delta_text
from ai_race_engineer.domains.common.analysis.pace.sectors import sector_deltas_s, sector_splits_s
from ai_race_engineer.domains.common.data.snapshot import Snapshot


class SectorComparison(Event):
    """When a class rival's new personal best beats the player's best, reports the per-sector deltas.
    _initialised   bool              first check after reset seeds silently
    _best_by_slot  dict[int, float]  s best lap of each class rival, by slot id
    """

    name = "sector_comparison"

    def __init__(self) -> None:
        """Start with no stored times.
        In:  -
        Out: None
        """
        self._initialised: bool = False
        self._best_by_slot: dict[int, float] = {}

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Callout for the fastest class rival who just set a best lap quicker than the player's.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "sector_comparison"; facts driver, lap_s, player_lap_s, d1_s, d2_s, d3_s
                                       (s, rival - player, negative = rival faster)
        """
        player = snapshot.player.vehicle
        rivals = [
            v
            for v in snapshot.vehicles
            if not v.is_player and v.class_name == player.class_name and v.best_lap_s is not None
        ]
        improvers = [
            v for v in rivals if v.slot_id not in self._best_by_slot or v.best_lap_s < self._best_by_slot[v.slot_id]
        ]
        self._best_by_slot.update({v.slot_id: v.best_lap_s for v in improvers})
        if not self._initialised:
            self._initialised = True
            return None
        if player.best_lap_s is None or player.best_sector1_s is None or player.best_sector2_cum_s is None:
            return None
        quicker = [
            v
            for v in improvers
            if v.best_lap_s < player.best_lap_s and v.best_sector1_s is not None and v.best_sector2_cum_s is not None
        ]
        if not quicker:
            return None
        rival = min(quicker, key=lambda v: v.best_lap_s)
        d1, d2, d3 = sector_deltas_s(
            sector_splits_s(rival.best_sector1_s, rival.best_sector2_cum_s, rival.best_lap_s),
            sector_splits_s(player.best_sector1_s, player.best_sector2_cum_s, player.best_lap_s),
        )
        return Callout(
            "sector_comparison",
            f"{rival.driver_name} is {player.best_lap_s - rival.best_lap_s:.2f} quicker: "
            f"S1 {delta_text(d1)}, S2 {delta_text(d2)}, S3 {delta_text(d3)}",
            Priority.INFO,
            {
                "driver": rival.driver_name,
                "lap_s": rival.best_lap_s,
                "player_lap_s": player.best_lap_s,
                "d1_s": d1,
                "d2_s": d2,
                "d3_s": d3,
            },
        )

    def reset(self) -> None:
        """Forget stored times; the next check seeds again.
        In:  -
        Out: None
        """
        self._initialised = False
        self._best_by_slot = {}
