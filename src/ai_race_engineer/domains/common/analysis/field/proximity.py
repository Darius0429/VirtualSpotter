import numpy as np
from numpy.typing import NDArray


def to_player_frame(
    player_xz_m: tuple[float, float], player_heading_rad: float, others_xz_m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Positions of other cars relative to the player's car.
    In:  player_xz_m         tuple[float, float]  m world (x, z)
         player_heading_rad  float                rad, forward = (sin h, cos h) in world (x, z)
         others_xz_m         NDArray[float64]     (n, 2) m world (x, z)
    Out:                     NDArray[float64]     (n, 2) m [lateral (+ right), longitudinal (+ ahead)]
    """
    d = others_xz_m - np.asarray(player_xz_m)
    sin_h, cos_h = np.sin(player_heading_rad), np.cos(player_heading_rad)
    lateral = d[:, 0] * cos_h - d[:, 1] * sin_h
    longitudinal = d[:, 0] * sin_h + d[:, 1] * cos_h
    return np.column_stack((lateral, longitudinal))
