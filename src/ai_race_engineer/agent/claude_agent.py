import base64
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from anthropic import Anthropic, APIError
from anthropic.types.beta import BetaMessage, BetaToolUseBlock

from ai_race_engineer.agent.base import Agent, answer_with_retries
from ai_race_engineer.agent.prompts import EFFORT_PROMPT, SYSTEM_PROMPT
from ai_race_engineer.agent.tool_schema import tool_schema
from ai_race_engineer.recorders.run_log import RunLog

EFFORT_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"effort": {"type": "string", "enum": ["low", "medium", "high"]}},
        "required": ["effort"],
        "additionalProperties": False,
    },
}
FALLBACK_BETA: str = "server-side-fallback-2026-07-01"

logger = logging.getLogger(__name__)


class ClaudeAgent(Agent):
    """One Claude agent that calls the race tools itself: a small model first picks the reasoning effort of the
    question, then the main model answers with that effort. The system prompt and the tool definitions are cached;
    every question stands alone. A tool that returns the path of a picture shows the picture to the model. Each
    round trip and tool call is written to the run log (stream "conversations").
    _client        Anthropic
    _tools         dict[str, Callable[..., object]]  tool name -> function
    _schemas       list[dict[str, object]]           strict tool definitions, in a fixed order
    _cache_ttl     str                               "5m" | "1h": how long the prompt stays cached
    _router_model  str                               model that picks the effort
    _model         str                               model that answers
    _max_tokens    Mapping[str, int]                 effort -> output token limit, thinking included
    _log           RunLog
    _context       Callable[[], str]                 one line that says where the driver is now
    """

    def __init__(
        self,
        client: Anthropic,
        tools: Sequence[Callable[..., object]],
        cache_ttl: str,
        router_model: str,
        model: str,
        max_tokens: Mapping[str, int],
        log: RunLog,
        context: Callable[[], str],
    ) -> None:
        """Store the client, the tools with their definitions and the models.
        In:  client        Anthropic
             tools         Sequence[Callable[..., object]]  typed functions with a docstring
             cache_ttl     str                              "5m" | "1h": how long the prompt stays cached
             router_model  str                              e.g. "claude-haiku-5-5"
             model         str                              e.g. "claude-sonnet-5-5"
             max_tokens    Mapping[str, int]                "low" | "medium" | "high" -> output token limit
             log           RunLog
             context       Callable[[], str]                one line that says where the driver is now
        Out:               None
        """
        self._client: Anthropic = client
        self._tools: dict[str, Callable[..., object]] = {tool.__name__: tool for tool in tools}
        self._schemas: list[dict[str, object]] = [tool_schema(tool) for tool in tools]
        self._cache_ttl: str = cache_ttl
        self._router_model: str = router_model
        self._model: str = model
        self._max_tokens: Mapping[str, int] = max_tokens
        self._log: RunLog = log
        self._context: Callable[[], str] = context

    def answer(self, question: str) -> str:
        """Answer one question, without earlier turns.
        In:  question  str
        Out:           str  short speech text; the configured error reply when the model or network fails
        """
        return answer_with_retries(lambda: self._respond(question))

    def warm(self) -> None:
        """Write the system prompt and the tools to the prompt cache before the first question, so that one is as
        fast as the later ones; a failure is logged and ignored (call once at start, off the main thread).
        In:  -
        Out: None
        """
        try:
            response = self._request("low", [{"role": "user", "content": "radio check"}], 0)
            self._log.write("conversations", {"event": "warm", "usage": response.usage.to_dict()})
        except APIError as error:
            logger.warning("prompt cache not warmed: %s", error)

    def _respond(self, question: str) -> str:
        """Pick the effort, then let the main model call tools until it answers.
        In:  question  str
        Out:           str  the model's final text; raises when it ends without text (refusal, token limit)
        """
        effort = self._effort(question)
        messages: list[dict[str, object]] = [{"role": "user", "content": f"{question}\n\n[{self._context()}]"}]
        while True:
            response = self._ask(effort, messages)
            if response.stop_reason != "tool_use":
                break
            results = [self._call(block) for block in response.content if block.type == "tool_use"]
            messages += [{"role": "assistant", "content": response.content}, {"role": "user", "content": results}]
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise RuntimeError(f"no answer: {response.stop_reason}")
        return text

    def _effort(self, question: str) -> str:
        """Reasoning effort the question needs, picked by the router model.
        In:  question  str
        Out:           str  "low" | "medium" | "high"
        """
        start_s = time.monotonic()
        response = self._client.messages.create(
            model=self._router_model,
            max_tokens=1024,
            system=EFFORT_PROMPT,
            messages=[{"role": "user", "content": question}],
            output_config={"effort": "low", "format": EFFORT_FORMAT},
        )
        effort = json.loads(next(block.text for block in response.content if block.type == "text"))["effort"]
        self._log.write("conversations", {"event": "effort", "effort": effort, "latency_s": time.monotonic() - start_s})
        return effort

    def _ask(self, effort: str, messages: list[dict[str, object]]) -> BetaMessage:
        """One round trip to the main model.
        In:  effort    str                      "low" | "medium" | "high"
             messages  list[dict[str, object]]  the question, then tool calls and results
        Out:           BetaMessage
        """
        start_s = time.monotonic()
        response = self._request(effort, messages, self._max_tokens[effort])
        self._log.write(
            "conversations",
            {"event": "model_end", "reply": response.to_dict(), "latency_s": time.monotonic() - start_s},
        )
        return response

    def _request(self, effort: str, messages: list[dict[str, object]], max_tokens: int) -> BetaMessage:
        """One request to the main model with the cached system prompt and tools.
        In:  effort      str                      "low" | "medium" | "high"
             messages    list[dict[str, object]]
             max_tokens  int                      output token limit; 0 only writes the prompt cache
        Out:             BetaMessage
        """
        cache = {"type": "ephemeral", "ttl": self._cache_ttl}
        return self._client.beta.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": cache}],
            tools=self._schemas,
            messages=messages,
            output_config={"effort": effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def _call(self, block: BetaToolUseBlock) -> dict[str, object]:
        """Run one tool call; a failing tool is reported to the model instead of ending the answer.
        In:  block  BetaToolUseBlock   tool name and arguments
        Out:        dict[str, object]  tool_result block
        """
        start_s = time.monotonic()
        try:
            output, failed = self._tools[block.name](**block.input), False
        except Exception as error:
            output, failed = repr(error), True
        self._log.write(
            "conversations",
            {
                "event": "tool_end",
                "tool": block.name,
                "input": block.input,
                "output": output,
                "failed": failed,
                "latency_s": time.monotonic() - start_s,
            },
        )
        content: object = json.dumps(output, default=str)
        if isinstance(output, Path):
            picture = base64.standard_b64encode(output.read_bytes()).decode()
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": picture}},
                {"type": "text", "text": f"saved as {output}"},
            ]
        return {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": failed}
