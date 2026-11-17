from ai_race_engineer.domains.lmu.plan_briefing import PlanBriefing

PLAN: dict[str, object] = {"plan_basis": "pre_race", "race_laps": 34, "stops": 1, "pit_at_end_of_lap": 22}


def test_plan_is_read_once_after_it_has_settled() -> None:
    """Nothing while the plan still moves; once it has been the same for the hold time it is read, one time until
    the next session."""
    plan = dict(PLAN, race_laps=35, stops=0)
    event = PlanBriefing(lambda: plan, lambda: {"leave_pits_with_percent": 53.6}, 20.0)
    assert event.check(None, 0.0) is None
    plan.update(PLAN)
    assert event.check(None, 15.0) is None
    assert event.check(None, 30.0) is None
    callout = event.check(None, 35.0)
    assert (callout.kind, callout.text) == (
        "plan_briefing",
        "34 laps. 1 stop, box end of lap 22, add energy to 54 percent.",
    )
    assert event.check(None, 60.0) is None
    event.reset()
    assert event.check(None, 70.0) is None
    assert event.check(None, 90.0) is not None


def test_plan_without_a_stop_and_plans_made_in_the_race() -> None:
    """A race without a stop says so; a plan that is no longer the pre-race one, or none, is not read."""
    no_stop = PlanBriefing(lambda: {**PLAN, "stops": 0}, dict, 0.0)
    assert no_stop.check(None, 0.0).text == "34 laps. No stop needed."
    assert PlanBriefing(lambda: {**PLAN, "plan_basis": "measured"}, dict, 0.0).check(None, 0.0) is None
    assert PlanBriefing(lambda: {"available": False}, dict, 0.0).check(None, 0.0) is None
