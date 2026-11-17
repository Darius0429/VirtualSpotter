from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.callouts.text import lap_time_text
from ai_race_engineer.domains.common.data.snapshot import Snapshot


class FastestLap(Event):
    """Announces a new overall session best lap (any car, any class) and the player's new personal best.
    _initialised     bool          first check after reset seeds silently
    _overall_best_s  float | None  s
    _player_best_s   float | None  s
    """

    name = "fastest_lap"

    def __init__(self) -> None:
        """Start with no stored times.
        In:  -
        Out: None
        """
        self._initialised: bool = False
        self._overall_best_s: float | None = None
        self._player_best_s: float | None = None

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Callout for a new overall best, else for a new player personal best.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "fastest_lap" (facts driver, class_name, lap_s, is_player) or
                                       "personal_best" (facts lap_s)
        """
        timed = [v for v in snapshot.vehicles if v.best_lap_s is not None]
        holder = min(timed, key=lambda v: v.best_lap_s) if timed else None
        player_best_s = snapshot.player.vehicle.best_lap_s
        if not self._initialised:
            self._initialised = True
            self._overall_best_s = holder.best_lap_s if holder else None
            self._player_best_s = player_best_s
            return None
        if holder is not None and (self._overall_best_s is None or holder.best_lap_s < self._overall_best_s):
            self._overall_best_s = holder.best_lap_s
            facts: dict[str, str | float | int] = {
                "driver": holder.driver_name,
                "class_name": holder.class_name,
                "lap_s": holder.best_lap_s,
                "is_player": int(holder.is_player),
            }
            if holder.is_player:
                self._player_best_s = player_best_s
                return Callout(
                    "fastest_lap",
                    f"You set the fastest time of the session with a {lap_time_text(holder.best_lap_s)}",
                    Priority.INFO,
                    facts,
                )
            return Callout(
                "fastest_lap",
                f"{holder.driver_name} set the fastest time of the session with a {lap_time_text(holder.best_lap_s)}",
                Priority.INFO,
                facts,
            )
        if player_best_s is not None and (self._player_best_s is None or player_best_s < self._player_best_s):
            self._player_best_s = player_best_s
            return Callout(
                "personal_best",
                f"You set your fastest lap with a {lap_time_text(player_best_s)}",
                Priority.INFO,
                {"lap_s": player_best_s},
            )
        return None

    def reset(self) -> None:
        """Forget stored times; the next check seeds again.
        In:  -
        Out: None
        """
        self._initialised = False
        self._overall_best_s = None
        self._player_best_s = None
