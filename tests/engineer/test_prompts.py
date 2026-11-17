from ai_race_engineer.domains.lmu.instructions import LMU_INSTRUCTIONS
from ai_race_engineer.engineer.playbook import PLAYBOOK
from ai_race_engineer.engineer.prompts import ROLE_PROMPT


def test_role_prompt_holds_the_playbook_before_the_talk_rules() -> None:
    """The playbook is joined into the role prompt, ahead of the rules on how to talk.
    In:  -
    Out: None
    """
    assert PLAYBOOK in ROLE_PROMPT
    assert ROLE_PROMPT.index(PLAYBOOK) < ROLE_PROMPT.index("# How to talk")


def test_role_prompt_knows_no_domain() -> None:
    """The engineer's half of the prompt names no game and nothing only one game has.
    In:  -
    Out: None
    """
    lowered = ROLE_PROMPT.lower()
    assert all(word not in lowered for word in ("lmu", "le mans ultimate", "rfactor", "virtual energy"))


def test_energy_rule_lives_in_the_lmu_instructions() -> None:
    """The rule that fuel means virtual energy belongs to the domain that has virtual energy.
    In:  -
    Out: None
    """
    assert "virtual energy" in LMU_INSTRUCTIONS
