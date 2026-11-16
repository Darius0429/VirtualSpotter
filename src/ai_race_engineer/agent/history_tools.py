from collections.abc import Callable

from ai_race_engineer.strategy.lap_history import LapHistory

UNAVAILABLE: dict[str, object] = {"available": False}


class HistoryTools:
    """Agent tool for how the gap to another car of the class moved over the laps of the race.
    _history  LapHistory  the facts of every finished race lap
    """

    def __init__(self, history: LapHistory) -> None:
        """Store the history.
        In:  history  LapHistory
        Out:          None
        """
        self._history: LapHistory = history

    def gap_history(self, driver: str, laps: int) -> dict[str, object]:
        """Lap by lap, the driver against one other car of the class over the last laps of the race: both lap
        times, both places and the gap between them at the end of each lap, with the change of the gap over the
        window. Use it for trends: whether the driver is catching or losing a car and by how much over how many
        laps, for example "gained 2 seconds in 5 laps". A positive gap means the other car is ahead on the road;
        gained_s is positive when the driver moved towards a car ahead or away from a car behind.
        In:  driver  str  driver of the other car, or part of the name; empty = the car directly ahead in class, or
                          the car behind when the driver leads the class
             laps    int  how many of the last laps; 0 = the whole race
        Out:         dict[str, object]  {driver, laps: [{lap, my_lap_s, their_lap_s, my_place, their_place, gap_s}],
                                        gap_change_s, gained_s, over_laps}; {"available": False} before a race lap
                                        is finished or when no car of the class matches
        """
        history = self._history.laps
        if not history:
            return UNAVAILABLE
        now = history[-1]["cars"]
        wanted = driver.strip().lower()
        mine = next(car for car in now if car["me"])
        target = next(
            (
                car
                for car in now
                if not car["me"]
                and (wanted in car["driver"].lower() if wanted else car["place"] == max(mine["place"] - 1, 1))
            ),
            None,
        )
        target = target or next((car for car in now if not wanted and car["place"] == mine["place"] + 1), None)
        if target is None:
            return UNAVAILABLE
        rows: list[dict[str, object]] = []
        for digest in history[-laps:] if laps else history:
            cars = {car["slot_id"]: car for car in digest["cars"]}
            them, me = cars.get(target["slot_id"]), next(car for car in digest["cars"] if car["me"])
            if them is not None:
                rows.append(
                    {
                        "lap": digest["lap"],
                        "my_lap_s": me["last_lap_s"],
                        "their_lap_s": them["last_lap_s"],
                        "my_place": me["place"],
                        "their_place": them["place"],
                        "gap_s": them["gap_to_me_s"],
                    }
                )
        change = round(rows[-1]["gap_s"] - rows[0]["gap_s"], 1)
        return {
            "driver": target["driver"],
            "laps": rows,
            "gap_change_s": change,
            "gained_s": -change,
            "over_laps": len(rows) - 1,
        }

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  gap_history
        """
        return [self.gap_history]
