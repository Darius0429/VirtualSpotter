import json
import urllib.request

from ai_race_engineer.telemetry.pit_estimate import PitEstimate, PitEstimateSource, pit_estimate


class LMUPitEstimateSource(PitEstimateSource):
    """Pit estimate from the Le Mans Ultimate REST API.
    _url        str
    _timeout_s  float  s
    """

    def __init__(self, url: str, timeout_s: float) -> None:
        """Set the endpoint.
        In:  url        str    pitstop-estimate endpoint of the game's REST API
             timeout_s  float  s
        Out:            None
        """
        self._url: str = url
        self._timeout_s: float = timeout_s

    def read(self) -> PitEstimate | None:
        """Ask the game for its pit estimate.
        In:  -
        Out: PitEstimate | None  None when the game is not reachable, the reply cannot be parsed or it is null
        """
        try:
            with urllib.request.urlopen(self._url, timeout=self._timeout_s) as response:
                return pit_estimate(json.load(response))
        except (OSError, ValueError, KeyError):
            return None
