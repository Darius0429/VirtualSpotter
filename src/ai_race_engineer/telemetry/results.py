import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

SESSION_KINDS: dict[str, str] = {"Practice": "practice", "Qualify": "qualify", "Warmup": "warmup", "Race": "race"}
TRACK_FIELDS: dict[str, str] = {
    "TrackVenue": "track_venue",
    "TrackCourse": "track_course",
    "TrackEvent": "track_event",
}
EVENT_KINDS: dict[str, str] = {
    "Incident": "incident",
    "Penalty": "penalty",
    "TrackLimits": "track_limits",
    "Sector": "sector",
}


@dataclass(frozen=True)
class ResultLap:
    """One lap of a driver in a results file.
    num             int           lap number, 1-based
    place           int           overall position at the end of the lap
    start_s         float | None  s session time at the start of the lap; None when the file has none
    lap_time_s      float | None  s; None for an untimed lap ("--.----")
    sector1_s       float | None  s
    sector2_s       float | None  s
    sector3_s       float | None  s
    compound_front  str           e.g. "Medium"
    compound_rear   str
    is_pit          bool          the lap ended in the pits (in-lap)
    """

    num: int
    place: int
    start_s: float | None
    lap_time_s: float | None
    sector1_s: float | None
    sector2_s: float | None
    sector3_s: float | None
    compound_front: str
    compound_rear: str
    is_pit: bool


@dataclass(frozen=True)
class ResultDriver:
    """One driver of a results file.
    name              str
    veh_name          str           vehicle name with team and number
    car_type          str           car model, e.g. "BMW M Hybrid V8"
    car_class         str
    car_number        str
    team_name         str
    is_player         bool          the file's isPlayer flag (set for every human driver of a multiplayer session)
    grid_place        int | None    overall grid position (race files only)
    class_grid_place  int | None    grid position in class (race files only)
    place             int           overall position
    class_place       int           position in class
    best_lap_s        float | None  s; None without a timed lap
    finish_time_s     float | None  s; None for a car that did not finish or outside a race
    laps              int           completed laps
    pitstops          int           count
    finish_status     str           e.g. "Finished Normally", "DNF"
    lap_list          tuple[ResultLap, ...]
    """

    name: str
    veh_name: str
    car_type: str
    car_class: str
    car_number: str
    team_name: str
    is_player: bool
    grid_place: int | None
    class_grid_place: int | None
    place: int
    class_place: int
    best_lap_s: float | None
    finish_time_s: float | None
    laps: int
    pitstops: int
    finish_status: str
    lap_list: tuple[ResultLap, ...]


@dataclass(frozen=True)
class ResultEvent:
    """One entry of the session's event stream.
    kind    str    "incident" | "penalty" | "track_limits" | "sector"
    time_s  float  s session time
    driver  str    the driver it is about ("" when the file names none)
    text    str    the game's message
    """

    kind: str
    time_s: float
    driver: str
    text: str


@dataclass(frozen=True)
class SessionResults:
    """One session of a results file the game writes when the session ends.
    track_venue     str
    track_course    str
    track_event     str
    track_length_m  float  m
    kind            str    "practice" | "qualify" | "warmup" | "race"
    started_unix    int    s unix time the session started
    race_time_min   float  minutes of the event's race (0 = no time limit)
    race_laps       int    laps of the event's race (0 = no lap limit)
    drivers         tuple[ResultDriver, ...]
    events          tuple[ResultEvent, ...]
    """

    track_venue: str
    track_course: str
    track_event: str
    track_length_m: float
    kind: str
    started_unix: int
    race_time_min: float
    race_laps: int
    drivers: tuple[ResultDriver, ...]
    events: tuple[ResultEvent, ...]


def parse_results(text: str) -> SessionResults | None:
    """Session results from the text of a results file; chat messages and score lines are not kept.
    In:  text  str                    XML document
    Out:       SessionResults | None  None for text that does not parse (half-written file) or has no session element
    """
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None
    results = root.find("RaceResults")
    session = None if results is None else next((e for e in results if _kind(e.tag)), None)
    if session is None:
        return None
    stream = session.find("Stream")
    return SessionResults(
        track_venue=results.findtext("TrackVenue", ""),
        track_course=results.findtext("TrackCourse", ""),
        track_event=results.findtext("TrackEvent", ""),
        track_length_m=float(results.findtext("TrackLength", "0")),
        kind=_kind(session.tag),
        started_unix=int(session.findtext("DateTime", "0")),
        race_time_min=float(results.findtext("RaceTime", "0")),
        race_laps=int(results.findtext("RaceLaps", "0")),
        drivers=tuple(_driver(e) for e in session.findall("Driver")),
        events=tuple(_event(e) for e in (stream if stream is not None else ()) if e.tag in EVENT_KINDS),
    )


def _kind(tag: str) -> str:
    """Session kind of an element tag.
    In:  tag  str  e.g. "Practice1", "Qualify", "Race"
    Out:      str  "practice" | "qualify" | "warmup" | "race"; "" for any other tag
    """
    return SESSION_KINDS.get(tag.rstrip("0123456789"), "")


def _driver(element: ElementTree.Element) -> ResultDriver:
    """One driver from its element.
    In:  element  ElementTree.Element  <Driver>
    Out:          ResultDriver
    """
    return ResultDriver(
        name=element.findtext("Name", ""),
        veh_name=element.findtext("VehName", ""),
        car_type=element.findtext("CarType", ""),
        car_class=element.findtext("CarClass", ""),
        car_number=element.findtext("CarNumber", ""),
        team_name=element.findtext("TeamName", ""),
        is_player=element.findtext("isPlayer") == "1",
        grid_place=_int(element.findtext("GridPos")),
        class_grid_place=_int(element.findtext("ClassGridPos")),
        place=int(element.findtext("Position", "0")),
        class_place=int(element.findtext("ClassPosition", "0")),
        best_lap_s=_time(element.findtext("BestLapTime")),
        finish_time_s=_time(element.findtext("FinishTime")),
        laps=int(element.findtext("Laps", "0")),
        pitstops=int(element.findtext("Pitstops", "0")),
        finish_status=element.findtext("FinishStatus", ""),
        lap_list=tuple(_lap(e) for e in element.findall("Lap")),
    )


def _lap(element: ElementTree.Element) -> ResultLap:
    """One lap from its element.
    In:  element  ElementTree.Element  <Lap>
    Out:          ResultLap
    """
    return ResultLap(
        num=int(element.get("num", "0")),
        place=int(element.get("p", "0")),
        start_s=_time(element.get("et")),
        lap_time_s=_time(element.text),
        sector1_s=_time(element.get("s1")),
        sector2_s=_time(element.get("s2")),
        sector3_s=_time(element.get("s3")),
        compound_front=element.get("fcompound", "").split(",")[-1],
        compound_rear=element.get("rcompound", "").split(",")[-1],
        is_pit=element.get("pit") == "1",
    )


def _event(element: ElementTree.Element) -> ResultEvent:
    """One stream entry from its element; an incident names its driver at the start of the message.
    In:  element  ElementTree.Element  <Incident>, <Penalty>, <TrackLimits> or <Sector>
    Out:          ResultEvent
    """
    text = (element.text or "").strip()
    driver = text.split("(")[0] if element.tag == "Incident" else ""
    return ResultEvent(
        kind=EVENT_KINDS[element.tag],
        time_s=float(element.get("et", "0")),
        driver=element.get("Driver", driver),
        text=text,
    )


def _time(text: str | None) -> float | None:
    """Time value of the file.
    In:  text  str | None    s as text; "--.----" or missing = no time
    Out:       float | None  s
    """
    return None if text is None or "-" in text or not text.strip() else float(text)


def _int(text: str | None) -> int | None:
    """Whole number of the file.
    In:  text  str | None  missing = None
    Out:       int | None
    """
    return None if text is None else int(text)
