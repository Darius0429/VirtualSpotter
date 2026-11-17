from collections.abc import Sequence

import pandas as pd

from ai_race_engineer.domains.common.data.snapshot import VehicleState


def standings(vehicles: Sequence[VehicleState]) -> pd.DataFrame:
    """Overall and class standings with class and session best laps.
    In:  vehicles  Sequence[VehicleState]  all cars
    Out:           pd.DataFrame  index slot_id; columns driver_name, class_name, place, position_in_class,
                                 best_lap_s (s, NaN if none), class_best_lap_s (s), session_best_lap_s (s)
    """
    table = pd.DataFrame(
        {
            "slot_id": [v.slot_id for v in vehicles],
            "driver_name": [v.driver_name for v in vehicles],
            "class_name": [v.class_name for v in vehicles],
            "place": [v.place for v in vehicles],
            "best_lap_s": pd.Series([v.best_lap_s for v in vehicles], dtype="float64"),
        }
    ).set_index("slot_id")
    by_class = table.groupby("class_name")
    table.insert(3, "position_in_class", by_class["place"].rank(method="first").astype(int))
    table["class_best_lap_s"] = by_class["best_lap_s"].transform("min")
    table["session_best_lap_s"] = table["best_lap_s"].min()
    return table
