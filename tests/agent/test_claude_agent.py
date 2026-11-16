import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_race_engineer.agent.claude_agent import ClaudeAgent
from ai_race_engineer.agent.segment_tools import SegmentTools
from ai_race_engineer.agent.tool_schema import tool_schema
from ai_race_engineer.recorders.corner_plot import CornerPlotter
from ai_race_engineer.recorders.run_log import RunLog
from ai_race_engineer.strategy.live_segments import LiveSegmentTracker
from ai_race_engineer.strategy.rival_lines import RivalLineTracker
from ai_race_engineer.strategy.segment_tracker import SegmentTracker
from tests.oval import oval_lap, with_rival
from tests.strategy.test_corner_card import LIMITS
from tests.strategy.test_corner_compare import FINDINGS

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


def _trackers() -> tuple[SegmentTracker, RivalLineTracker, LiveSegmentTracker]:
    """Segment tracker after four laps of the oval: lap 2 (40.0 s, braked 20 m earlier), lap 3 (39.5 s) and
    lap 4 (40.5 s) are analysed. Rival One drives the same laps 3 m further in x and 0.5 s faster.
    In:  -
    Out: tuple[SegmentTracker, RivalLineTracker, LiveSegmentTracker]
    """
    rivals = RivalLineTracker()
    live = LiveSegmentTracker(lambda: tracker.segments)
    tracker = SegmentTracker(
        lambda track: (),
        lambda track, segments: None,
        lambda track, car: None,
        lambda track, car, lap_time_s, samples: None,
        keep_laps=30,
        limits=LIMITS,
        step_m=5.0,
        smooth_m=55.0,
        max_radius_m=400.0,
        min_turn_deg=18.0,
        entry_pad_m=100.0,
        exit_pad_m=50.0,
        merge_gap_m=50.0,
    )
    for completed_laps, last_lap_s, brake_zone_m in (
        (0, None, 100.0),
        (1, 41.0, 120.0),
        (2, 40.0, 100.0),
        (3, 39.5, 100.0),
        (4, 40.5, 100.0),
    ):
        rival_s = None if last_lap_s is None else last_lap_s - 0.5
        lap = oval_lap(completed_laps, last_lap_s, brake_zone_m=brake_zone_m)
        for snapshot in with_rival(lap, "Rival One", rival_s, shift_x_m=3.0):
            tracker.update(snapshot)
            rivals.update(snapshot)
            live.update(snapshot)
    return tracker, rivals, live


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


def _tools(tmp_path: Path) -> SegmentTools:
    """Segment tools on the simulated laps, drawing into the test folder.
    In:  tmp_path  Path
    Out:           SegmentTools
    """
    tracker, rivals, live = _trackers()
    return SegmentTools(tracker, rivals, live, CornerPlotter(tmp_path, 60.0, 0.05, 0.05), FINDINGS, 150.0)


@pytest.mark.parametrize(
    ("effort", "question", "tool", "arguments"),
    [
        ("low", "How are my tyres?", "car_health", {}),
        ("medium", "Where am I losing time in turn 1?", "corner_analysis", {"corner": "turn 1", "lap": 2}),
        ("high", "Where can I find time over the lap?", "lap_losses", {"lap": 2}),
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
    assert strict == {
        "lap_losses": True,
        "this_lap": False,
        "live_corners": True,
        "corner_analysis": True,
        "corner_plot": True,
        "rival_comparison": True,
        "car_health": False,
    }
    result = main.calls[1]["messages"][2]["content"][0]
    assert (result["type"], result["tool_use_id"], result["is_error"]) == ("tool_result", "call_1", False)
    output = json.loads(result["content"])
    if tool == "car_health":
        assert (output["compound_front"], output["wheels"], output["track_temp_c"]) == ("Medium", {}, 30.0)
    elif tool == "corner_analysis":
        assert (output["lap"], output["best_lap"], output["name"]) == (2, 3, "turn 1")
        assert output["vs_best"]["findings"][0].endswith(" m earlier")
        assert output["vs_best"]["lost_s"] > 0.0
        assert output["entry"]["brake_point_m"] < output["mid"]["min_speed_m"]
    else:
        assert (output["lap"], output["best_lap"], output["analysed_laps"]) == (2, 3, [2, 3, 4])
        assert output["theoretical_best_s"] <= output["best_lap_time_s"]
        assert output["segments"][0]["lost_s"] >= output["segments"][-1]["lost_s"]
        assert {s["name"] for s in output["segments"]} >= {"turn 1", "turn 2", "straight 1"}


def test_corner_plot_is_saved_and_shown_to_the_model(tmp_path: Path) -> None:
    """A tool that returns a picture file sends the picture itself to the model, with where it was saved."""
    agent, _, main = _agent(
        tmp_path,
        "medium",
        _reply("tool_use", _tool_use("corner_plot", corner="Turn 1", lap=2, against="", their_last_lap=False)),
        _reply("end_turn", _text("Brake later.")),
    )
    assert agent.answer("Why am I slow in turn 1?") == "Brake later."
    picture, note = main.calls[1]["messages"][2]["content"][0]["content"]
    assert (picture["type"], picture["source"]["media_type"]) == ("image", "image/png")
    assert (tmp_path / "lap 2 turn 1 against lap 3.png").read_bytes().startswith(b"\x89PNG")
    assert note["text"].endswith("lap 2 turn 1 against lap 3.png")


def test_unknown_corner_or_lap_is_unavailable(tmp_path: Path) -> None:
    """A segment the track does not have or a lap that was not analysed gives no data."""
    tools = _tools(tmp_path)
    assert tools.corner_analysis("turn 9", 0) == {"available": False}
    assert tools.corner_plot("turn 1", 7, "", False) == {"available": False}
    assert tools.corner_plot("turn 1", 0, "nobody", False) == {"available": False}
    assert tools.rival_comparison("nobody", "", False) == {"available": False, "drivers": ["Rival One"]}
    assert tools.lap_losses(7) == {"available": False}


def test_failing_tool_is_reported_to_the_model(tmp_path: Path) -> None:
    """A tool call that raises goes back as an error result and the model still answers."""
    agent, _, main = _agent(
        tmp_path, "low", _reply("tool_use", _tool_use("lap_losses", lap=0, extra=1)), _reply("end_turn", _text("No."))
    )
    assert agent.answer("Turn 1?") == "No."
    assert main.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_reply_without_text_gives_the_error_reply(tmp_path: Path) -> None:
    """A refusal has no text: the answer is the configured error reply (empty by default)."""
    agent, _, _ = _agent(tmp_path, "low", _reply("refusal"))
    assert agent.answer("Hello?") == ""


def test_schema_requires_every_typed_argument(tmp_path: Path) -> None:
    """A typed function becomes a strict tool with all its arguments required."""
    schema = tool_schema(_tools(tmp_path).corner_analysis)
    assert schema["name"] == "corner_analysis"
    assert schema["description"].startswith("Everything the driver did in one corner")
    assert schema["input_schema"] == {
        "type": "object",
        "properties": {"corner": {"type": "string"}, "lap": {"type": "integer"}},
        "required": ["corner", "lap"],
        "additionalProperties": False,
    }


def test_rival_comparison_ranks_the_lap_and_details_a_corner(tmp_path: Path) -> None:
    """Against the fastest car of the class: the segments by time lost, and for one corner both cars' figures, the
    difference and the line; the plot against the rival is saved under its name."""
    tools = _tools(tmp_path)
    lap = tools.rival_comparison("", "", False)
    assert (lap["driver"], lap["their_lap_time_s"], lap["my_lap"]) == ("Rival One", 39.0, 3)
    assert sum(s["lost_s"] for s in lap["segments"]) == pytest.approx(0.5, abs=0.02)
    assert lap["segments"][0]["lost_s"] >= lap["segments"][-1]["lost_s"]
    corner = tools.rival_comparison("rival", "Turn 1", False)
    assert corner["name"] == "turn 1"
    assert corner["difference"]["min_kph"] == pytest.approx(0.0, abs=0.1)
    assert corner["me"]["slowing_point_m"] == pytest.approx(corner["them"]["slowing_point_m"], abs=4.0)
    assert abs(corner["line_wider_m"]) == pytest.approx(3.0, abs=0.5)
    assert tools.corner_plot("turn 1", 3, "rival", False).name == "lap 3 turn 1 against Rival One best lap.png"
    assert tools.corner_plot("turn 1", 0, "rival", True).name == "lap 4 turn 1 against Rival One last lap.png"


def test_rival_comparison_of_the_last_laps(tmp_path: Path) -> None:
    """With last_lap the driver's last analysed lap is compared with the other car's last full lap."""
    lap = _tools(tmp_path).rival_comparison("", "", True)
    assert (lap["laps_compared"], lap["my_lap"]) == ("last", 4)
    assert (lap["my_lap_time_s"], lap["their_lap_time_s"]) == (40.5, 40.0)
    assert sum(s["lost_s"] for s in lap["segments"]) == pytest.approx(0.5, abs=0.02)


def test_warming_writes_the_cache_without_asking_for_output(tmp_path: Path) -> None:
    """Warming sends the cached system prompt and the tools with a zero output limit."""
    agent, router, main = _agent(tmp_path, "low", SimpleNamespace(usage=SimpleNamespace(to_dict=dict)))
    agent.warm()
    assert router.calls == []
    assert main.calls[0]["max_tokens"] == 0
    assert main.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert len(main.calls[0]["tools"]) == 7


def test_best_lap_and_class_fastest_comparisons(tmp_path: Path) -> None:
    """Lap -1 is the driver's best lap; the lap ranking also says where it loses to the fastest lap of the class,
    and a corner also gives the fastest pass of the class."""
    tools = _tools(tmp_path)
    losses = tools.lap_losses(-1)
    assert (losses["lap"], losses["best_lap"]) == (3, 3)
    fastest = losses["against_fastest_in_class"]
    assert (fastest["driver"], fastest["their_lap_time_s"], len(fastest["segments"])) == ("Rival One", 39.0, 5)
    assert fastest["segments"][0]["lost_s"] >= fastest["segments"][-1]["lost_s"]
    corner = tools.corner_analysis("turn 1", -1)
    assert corner["fastest_pass_in_class"]["driver"] == "Rival One"
    assert abs(corner["fastest_pass_in_class"]["line_wider_m"]) == pytest.approx(3.0, abs=0.5)


def test_lap_in_progress_and_live_corners(tmp_path: Path) -> None:
    """Part of a new lap braked 20 m earlier: the corner already left is on this lap with what was different, and
    the driver's latest pass of each segment is set against the rival's."""
    tracker, rivals, live = _trackers()
    tools = SegmentTools(tracker, rivals, live, CornerPlotter(tmp_path, 60.0, 0.05, 0.05), FINDINGS, 150.0)
    assert [s["name"] for s in tools.this_lap()["segments"]] == ["turn 2", "straight 2", "turn 1", "straight 1"]
    lap = with_rival(oval_lap(5, 40.5, brake_zone_m=120.0), "Rival One", 40.0, shift_x_m=3.0)
    for snapshot in lap[: len(lap) * 2 // 3]:
        tracker.update(snapshot)
        live.update(snapshot)
    now = tools.this_lap()
    assert [s["name"] for s in now["segments"]] == ["straight 2", "turn 1", "straight 1"]
    assert now["segments"][1]["findings"][0].endswith(" m earlier")
    assert now["lost_so_far_s"] == pytest.approx(sum(s["lost_s"] for s in now["segments"]), abs=0.02)
    corners = tools.live_corners("")
    assert corners["driver"] == "Rival One"
    assert {"turn 1", "straight 1"} <= {s["name"] for s in corners["segments"]}
    assert all(abs(s["lost_s"]) < 0.1 for s in corners["segments"])
    assert tools.live_corners("nobody") == {"available": False}
