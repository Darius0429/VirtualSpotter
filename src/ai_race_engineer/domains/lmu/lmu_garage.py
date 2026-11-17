import http.client
import json
import urllib.request

from ai_race_engineer.domains.common.data.garage import GarageInfo, GarageSource, garage_info


class LMUGarageSource(GarageSource):
    """Garage figures from the Le Mans Ultimate REST API.
    _url        str
    _timeout_s  float  s
    """

    def __init__(self, url: str, timeout_s: float) -> None:
        """Set the endpoint.
        In:  url        str    player-garage-data endpoint of the game's REST API
             timeout_s  float  s
        Out:            None
        """
        self._url: str = url
        self._timeout_s: float = timeout_s

    def read(self) -> GarageInfo | None:
        """Ask the game for the player's garage data.
        In:  -
        Out: GarageInfo | None  None when the game is not reachable, the player has no car in a session (HTTP 404),
                                the connection breaks off or the reply cannot be parsed
        """
        try:
            with urllib.request.urlopen(self._url, timeout=self._timeout_s) as response:
                return garage_info(json.load(response))
        except (OSError, ValueError, AttributeError, http.client.HTTPException):
            return None
