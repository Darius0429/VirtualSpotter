import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_race_engineer.agent.claude_agent import ClaudeAgent
from ai_race_engineer.agent.store_tools import StoreTools, _named
from ai_race_engineer.agent.tool_schema import tool_schema
from ai_race_engineer.recorders.corner_plot import CornerPlotter
from ai_race_engineer.recorders.run_log import RunLog
from tests.recorders.test_segment_recorder import Run, run

MAX_TOKENS: dict[str, int] = {"low": 4096, "medium": 4096, "high": 16384}


class _Messages:
    """Stand-in for a messages endpoint that gives prepared replies.
    calls     list[dict[str, object]]  arguments of every request
    _replies  list[SimpleNamespace]    replies left
    """

    def __init__(self, replies: list[SimpleNamespace]) -> None:
        """Hold the replies.
        In:  replies  list[SimpleNamespace]
        Out:          None
        """
        self.calls: list[dict[str, object]] = []
        self._replies: list[SimpleNamespace] = replies

    def create(self, **request: object) -> SimpleNamespace:
        """Record the request and give the next reply.
        In:  request  object
        Out:          SimpleNamespace
        """
        self.calls.append(request)
        return self._replies.pop(0)


def _reply(stop_reason: str, *content: SimpleNamespace) -> SimpleNamespace:
    """A model reply.
    In:  stop_reason  str
         content      SimpleNamespace  content blocks
    Out:              SimpleNamespace
    """
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), to_dict=dict)


def _text(text: str) -> SimpleNamespace:
    """A text block.
    In:  text  str
    Out:       SimpleNamespace
    """
    return SimpleNamespace(type="text", text=text)


def _tool_use(name: str, **arguments: object) -> SimpleNamespace:
    """A tool call block.
    In:  name       str
         arguments  object
    Out:            SimpleNamespace
    """
    return SimpleNamespace(type="tool_use", id="call_1", name=name, input=arguments)


def _agent(tmp_path: Path, effort: str, *replies: SimpleNamespace) -> tuple[ClaudeAgent, _Messages, _Messages]:
    """Agent on the segment tools with a router that picks the effort and a main model that gives the replies.
    In:  tmp_path  Path
         effort    str              "low" | "medium" | "high"
         replies   SimpleNamespace  replies of the main model, in order
    Out:           tuple[ClaudeAgent, _Messages, _Messages]  agent, router endpoint, main endpoint
    """
    router = _Messages([_reply("end_turn", _text(json.dumps({"effort": effort})))])
    main = _Messages(list(replies))
    client = SimpleNamespace(messages=router, beta=SimpleNamespace(messages=main))
    agent = ClaudeAgent(
        client,
        _tools(tmp_path).tools(),
        cache_ttl="1h",
        router_model="claude-haiku-5-5",
        model="claude-sonnet-5-5",
        max_tokens=MAX_TOKENS,
        log=RunLog(tmp_path, False, lambda: 0.0),
        context=lambda: "Now: race, on lap 5, at turn 1.",
    )
    return agent, router, main


def _tools(tmp_path: Path, driven: Run | None = None) -> StoreTools:
    """Store tools on the simulated laps, drawing into the test folder.
    In:  tmp_path  Path
         driven    Run | None  the run to read; None = five laps of the oval
    Out:           StoreTools
    """
    driven = driven or run(tmp_path)
    plotter = CornerPlotter(tmp_path, 60.0, 0.05, 0.05)
    return StoreTools(driven.store, driven.tracker, plotter, 150.0)


@pytest.mark.parametrize(
    ("effort", "question", "tool", "arguments"),
    [
        ("low", "How are my tyres?", "car_health", {}),
        (
            "medium",
            "Where am I losing time in turn 1?",
            "lap_data",
            {"car": "", "lap": "4", "part": "turn 1", "against": "car"},
        ),
        (
            "high",
            "Where can I find time over the lap?",
            "lap_data",
            {"car": "", "lap": "last", "part": "", "against": "car"},
        ),
    ],
)
def test_question_is_routed_answered_with_a_tool_and_spoken(
    tmp_path: Path, effort: str, question: str, tool: str, arguments: dict[str, object]
) -> None:
    """The router's effort sets the effort and the token limit of the main model, the tool the model asks for is
    run on the simulated laps and its JSON goes back, and the final text is the answer."""
    agent, router, main = _agent(
        tmp_path, effort, _reply("tool_use", _tool_use(tool, **arguments)), _reply("end_turn", _text("Copy."))
    )
    assert agent.answer(question) == "Copy."
    assert router.calls[0]["model"] == "claude-haiku-5-5"
    assert router.calls[0]["messages"] == [{"role": "user", "content": question}]
    assert main.calls[0]["messages"][0]["content"].split("\n") == [question, "", "[Now: race, on lap 5, at turn 1.]"]
    assert router.calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert [call["output_config"] for call in main.calls] == [{"effort": effort}] * 2
    assert main.calls[0]["max_tokens"] == MAX_TOKENS[effort]
    assert main.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    strict = {schema["name"]: schema["strict"] for schema in main.calls[0]["tools"]}
    assert strict == {"lap_data": True, "corner_plot": True, "corner_lines": True, "car_health": False}
    result = main.calls[1]["messages"][2]["content"][0]
    assert (result["type"], result["tool_use_id"], result["is_error"]) == ("tool_result", "call_1", False)
    output = json.loads(result["content"])
    if tool == "car_health":
        assert (output["compound_front"], output["wheels"], output["track_temp_c"]) == ("Medium", {}, 30.0)
    elif arguments["part"]:
        assert (output["lap"], output["name"], output["in_progress"]) == (4, "turn 1", False)
        assert (output["vs"]["reference"], output["vs"]["reference_lap_s"]) == ("lap 3", 39.5)
        assert output["entry"]["brake_point_m"] < output["mid"]["min_speed_m"]
        assert set(output["other_references"]) == {"class", "track"}
    else:
        assert (output["lap"], output["reference"], output["time_s"], output["lost_s"]) == (4, "lap 3", 40.5, 1.0)
        assert [sector["name"] for sector in output["sectors"]] == ["sector 1", "sector 2", "sector 3"]
        assert output["theoretical_best_s"] <= 39.5
        assert output["segments"][0]["lost_s"] >= output["segments"][-1]["lost_s"]
        assert {s["name"] for s in output["segments"]} == {"straight 1", "turn 1", "straight 2", "turn 2", "straight 3"}


def test_corner_plot_is_saved_and_shown_to_the_model(tmp_path: Path) -> None:
    """A tool that returns a picture file sends the picture itself to the model, with where it was saved."""
    agent, _, main = _agent(
        tmp_path,
        "medium",
        _reply("tool_use", _tool_use("corner_plot", corner="Turn 1", lap="4", against="car")),
        _reply("end_turn", _text("Brake later.")),
    )
    assert agent.answer("Why am I slow in turn 1?") == "Brake later."
    picture, note = main.calls[1]["messages"][2]["content"][0]["content"]
    assert (picture["type"], picture["source"]["media_type"]) == ("image", "image/png")
    assert (tmp_path / "lap 4 turn 1 against lap 3.png").read_bytes().startswith(b"\x89PNG")
    assert note["text"].endswith("lap 4 turn 1 against lap 3.png")


def test_unknown_car_lap_or_part_is_unavailable(tmp_path: Path) -> None:
    """A segment the track does not have, a lap that was not recorded, a car nobody drives or a lap that had no
    reference to be drawn against gives no data."""
    tools = _tools(tmp_path)
    assert tools.lap_data("", "4", "turn 9", "car") == {"available": False}
    assert tools.lap_data("", "7", "", "car") == {"available": False}
    assert tools.lap_data("nobody", "last", "", "me") == {"available": False}
    assert tools.lap_data("", "last", "", "nobody") == {"available": False}
    assert tools.corner_plot("turn 1", "7", "car") == {"available": False}
    assert tools.corner_plot("turn 1", "4", "nobody") == {"available": False}
    assert tools.corner_plot("turn 1", "2", "car") == {"available": False}
    assert tools.corner_lines("nobody", "turn 1", "best") == {"available": False}
    assert tools.corner_lines("rival", "turn 9", "best") == {"available": False}


def test_failing_tool_is_reported_to_the_model(tmp_path: Path) -> None:
    """A tool call that raises goes back as an error result and the model still answers."""
    agent, _, main = _agent(
        tmp_path, "low", _reply("tool_use", _tool_use("lap_data", car="", extra=1)), _reply("end_turn", _text("No."))
    )
    assert agent.answer("Turn 1?") == "No."
    assert main.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_reply_without_text_gives_the_error_reply(tmp_path: Path) -> None:
    """A refusal has no text: the answer is the configured error reply (empty by default)."""
    agent, _, _ = _agent(tmp_path, "low", _reply("refusal"))
    assert agent.answer("Hello?") == ""


def test_schema_requires_every_typed_argument(tmp_path: Path) -> None:
    """A typed function becomes a strict tool with all its arguments required."""
    schema = tool_schema(_tools(tmp_path).lap_data)
    assert schema["name"] == "lap_data"
    assert schema["description"].startswith("Times and time lost of any car")
    names = ["car", "lap", "part", "against"]
    assert schema["input_schema"] == {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        "required": names,
        "additionalProperties": False,
    }


def test_driver_against_another_car_and_the_other_way_round(tmp_path: Path) -> None:
    """Best against best and last against last come from the recorded times of both cars, per sector and per
    segment; asked for the other car, the signs turn round. One corner in detail and its plot come from the lines."""
    tools = _tools(tmp_path)
    best = tools.lap_data("", "best", "", "rival")
    assert (best["against"], best["lap"], best["their_lap"]) == ("Rival One", 3, 3)
    assert (best["time_s"], best["their_time_s"], best["lost_s"]) == (39.5, 39.0, 0.5)
    assert best["segments"][0]["lost_s"] >= best["segments"][-1]["lost_s"]
    last = tools.lap_data("", "last", "", "Rival")
    assert (last["lap"], last["time_s"], last["their_time_s"], last["lost_s"]) == (4, 40.5, 40.0, 0.5)
    assert len(last["segments"]) == 5
    assert all(abs(segment["lost_s"]) < 0.2 for segment in last["segments"])
    assert tools.lap_data("rival", "last", "", "me")["lost_s"] == -0.5
    sector = tools.lap_data("", "last", "sector 1", "rival")
    assert {segment["name"] for segment in sector["segments"]} == {"straight 1", "turn 1"}
    corner = tools.corner_lines("", "Turn 1", "best")
    assert (corner["name"], corner["my_lap"], corner["driver"], corner["their_lap"]) == ("turn 1", 3, "Rival One", 3)
    assert tools.corner_lines("rival", "turn 1", "4")["their_lap_time_s"] == 40.0
    assert corner["difference"]["min_kph"] == pytest.approx(0.0, abs=0.1)
    assert abs(corner["line_wider_m"]) == pytest.approx(3.0, abs=0.5)
    assert tools.corner_plot("turn 1", "4", "rival").name == "lap 4 turn 1 against Rival One lap 4.png"
    assert tools.corner_plot("turn 1", "4", "fastest").name == "lap 4 turn 1 against Rival One lap 4.png"
    assert tools.corner_plot("turn 1", "best", "rival").name == "lap 3 turn 1 against Rival One lap 3.png"
    assert tools.corner_plot("turn 1", "last", "my best").name == "lap 4 turn 1 against my lap 3.png"
    assert tools.corner_plot("turn 1", "4", "my lap 2").name == "lap 4 turn 1 against my lap 2.png"
    fastest = tools.lap_data("", "best", "", "fastest")
    assert (fastest["against"], fastest["lap"], fastest["their_lap"], fastest["lost_s"]) == ("Rival One", 3, 3, 0.5)


def test_warming_writes_the_cache_without_asking_for_output(tmp_path: Path) -> None:
    """Warming sends the cached system prompt and the tools with a zero output limit."""
    agent, router, main = _agent(tmp_path, "low", SimpleNamespace(usage=SimpleNamespace(to_dict=dict)))
    agent.warm()
    assert router.calls == []
    assert main.calls[0]["max_tokens"] == 0
    assert main.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert len(main.calls[0]["tools"]) == 4


def test_class_fastest_and_sector_comparisons(tmp_path: Path) -> None:
    """Against "class" a corner is compared with the fastest lap of the class, from the line and the speed of that
    car; a sector gives its time, the time lost and its segments."""
    tools = _tools(tmp_path)
    corner = tools.lap_data("", "best", "turn 1", "class")
    assert (corner["lap"], corner["vs"]["reference"]) == (3, "Rival One")
    assert abs(corner["vs"]["line_wider_m"]) == pytest.approx(3.0, abs=0.5)
    assert corner["other_references"]["car"]["reference"] == "lap 2"
    sector = tools.lap_data("", "4", "sector 1", "car")
    assert (sector["name"], sector["reference"]) == ("sector 1", "lap 3")
    assert {segment["name"] for segment in sector["segments"]} == {"straight 1", "turn 1"}
    assert sector["lost_s"] == pytest.approx(sum(s["lost_s"] for s in sector["segments"]), abs=0.3)


def test_lap_in_progress_against_the_best_lap_and_against_another_car(tmp_path: Path) -> None:
    """Part of a new lap braked 20 m earlier: the segments already left are there, latest first, with what was
    different, the sector that is complete, the plot of the corner just driven, and the same segments against the
    other car's latest pass of each."""
    driven = run(tmp_path)
    tools = _tools(tmp_path, driven)
    assert (tools.lap_data("", "now", "", "car")["lap"], tools.lap_data("", "last", "", "car")["lap"]) == (5, 4)
    driven.drive(5, 40.5, 120.0, share=0.66)
    now = tools.lap_data("", "now", "", "car")
    assert (now["lap"], now["in_progress"], now["time_s"]) == (6, True, None)
    assert [s["name"] for s in now["segments"]] == ["straight 2", "turn 1", "straight 1"]
    assert now["segments"][1]["findings"][0].endswith(" m earlier")
    assert now["lost_s"] == pytest.approx(sum(s["lost_s"] for s in now["segments"]), abs=0.02)
    assert [sector["name"] for sector in now["sectors"]] == ["sector 1"]
    assert tools.corner_plot("turn 1", "now", "car").name == "lap 6 turn 1 against lap 3.png"
    fight = tools.lap_data("", "now", "", "rival")
    assert (fight["against"], fight["their_lap"]) == ("Rival One", None)
    assert [s["name"] for s in fight["segments"]] == ["straight 2", "turn 1", "straight 1"]
    assert all(abs(s["lost_s"]) < 0.1 for s in fight["segments"])


def test_corner_made_of_two_turns_is_found_by_either_number() -> None:
    """A segment that holds two official turns answers to each of them, and to nothing else."""
    assert _named("turn 9 and 10", "turn 10") and _named("turn 9 and 10", "turn 9")
    assert _named("arnage", "arnage")
    assert not _named("turn 9 and 10", "turn 1") and not _named("straight 10", "turn 10")
