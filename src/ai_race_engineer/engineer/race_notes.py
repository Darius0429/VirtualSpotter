import json
import logging
import time
from collections.abc import Sequence

from anthropic import Anthropic, APIError

from ai_race_engineer.engineer.prompts import RACE_NOTES_PROMPT, RACE_SUMMARY_PROMPT
from ai_race_engineer.domains.base import RunLog

logger = logging.getLogger(__name__)

NOTES_PER_LAP: int = 2

NOTES_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"notes": {"type": "array", "items": {"type": "string"}}},
        "required": ["notes"],
        "additionalProperties": False,
    },
}


class RaceNoteWriter:
    """Writes the engineer's notes of the race with a model: after each lap the few things that matter to the
    driver's race, and at the flag a spoken summary. Every call is written to the run log (stream "race_notes"); a
    failed call gives no note.
    _client         Anthropic
    _notes_model    str        model that picks what matters on a lap
    _summary_model  str        model that sums the race up
    _log            RunLog
    """

    def __init__(self, client: Anthropic, notes_model: str, summary_model: str, log: RunLog) -> None:
        """Store the client, the models and the log.
        In:  client         Anthropic
             notes_model    str        e.g. "claude-haiku-5-5"
             summary_model  str        e.g. "claude-sonnet-5-5"
             log            RunLog
        Out:                None
        """
        self._client: Anthropic = client
        self._notes_model: str = notes_model
        self._summary_model: str = summary_model
        self._log: RunLog = log

    def notes(self, digest: dict[str, object], story: Sequence[str]) -> list[str]:
        """The things of one lap worth remembering for the driver's race.
        In:  digest  dict[str, object]  the facts of the lap (lap_digest)
             story   Sequence[str]      the notes written so far, oldest first
        Out:         list[str]          at most two short notes, most laps none; empty when the model or network
                                fails
        """
        text = self._ask(
            self._notes_model,
            RACE_NOTES_PROMPT,
            json.dumps({"notes_so_far": list(story), "lap_facts": digest}),
            {"effort": "low", "format": NOTES_FORMAT},
            {"lap": digest["lap"], "digest": digest},
        )
        return list(json.loads(text)["notes"])[:NOTES_PER_LAP] if text else []

    def summary(self, story: Sequence[str], timeline: Sequence[str], laps: Sequence[dict[str, object]]) -> str:
        """Spoken debrief of the driver's race.
        In:  story     Sequence[str]                the notes of the race, oldest first
             timeline  Sequence[str]                the timeline entries of the race, oldest first
             laps      Sequence[dict[str, object]]  the facts of every finished race lap (lap_digest), oldest first
        Out:           str                          spoken sentences; "" when the model or network fails
        """
        mine = [next(car for car in digest["cars"] if car["me"]) for digest in laps]
        figures = {
            "my_laps": [
                {"lap": digest["lap"], "lap_s": car["last_lap_s"], "place": car["place"]}
                for digest, car in zip(laps, mine, strict=True)
            ],
            "final_class_order": laps[-1]["cars"],
        }
        return self._ask(
            self._summary_model,
            RACE_SUMMARY_PROMPT,
            json.dumps({"notes": list(story), "timeline": list(timeline), **figures}),
            {"effort": "low"},
            {"lap": "finish"},
        )

    def _ask(self, model: str, system: str, content: str, output: dict[str, object], record: dict[str, object]) -> str:
        """One model call, logged.
        In:  model    str
             system   str                system prompt
             content  str                the user message
             output   dict[str, object]  output_config of the request
             record   dict[str, object]  what identifies the call in the log
        Out:          str                the model's text; "" when the model or network fails
        """
        start_s = time.monotonic()
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config=output,
            )
        except APIError as error:
            logger.warning("race notes call failed: %s", error)
            return ""
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage.to_dict()
        self._log.write(
            "race_notes", {**record, "reply": text, "usage": usage, "latency_s": time.monotonic() - start_s}
        )
        return text
