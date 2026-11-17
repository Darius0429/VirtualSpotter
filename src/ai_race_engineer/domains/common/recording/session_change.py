from ai_race_engineer.domains.common.data.frame import SessionInfo


def is_new_session(prev: SessionInfo | None, cur: SessionInfo, prev_time_s: float, time_s: float) -> bool:
    """Whether a frame starts a new recorded session.
    In:  prev         SessionInfo | None  session of the previous frame (None before the first frame)
         cur          SessionInfo         session of this frame
         prev_time_s  float               s telemetry clock of the previous frame
         time_s       float               s telemetry clock of this frame
    Out:              bool  True on the first frame, a track/vehicle/session type/session number change,
                            or the clock going backwards (restart)
    """
    return (
        prev is None
        or (cur.track_name, cur.vehicle_name, cur.session_type, cur.session_number)
        != (prev.track_name, prev.vehicle_name, prev.session_type, prev.session_number)
        or time_s < prev_time_s
    )
