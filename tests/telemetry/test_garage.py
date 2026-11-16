import pytest

from ai_race_engineer.telemetry.garage import GarageInfo, garage_info, merge_garage


def _data(energy: str, ratio: str) -> dict[str, dict[str, str]]:
    """Garage JSON object with the two settings that are read.
    In:  energy  str  stringValue of VM_VIRTUAL_ENERGY
         ratio   str  stringValue of VM_FUEL_LEVEL
    Out:         dict[str, dict[str, str]]
    """
    return {"VM_VIRTUAL_ENERGY": {"stringValue": energy}, "VM_FUEL_LEVEL": {"stringValue": ratio}}


def test_energy_per_lap_from_the_laps_text() -> None:
    """100 % for 36.2 laps is 1 / 36.2 per lap; 5 % for 1.7 laps is 0.05 / 1.7."""
    assert garage_info(_data("100% (36.2 laps)", "0.86")).energy_per_lap == pytest.approx(1 / 36.2)
    assert garage_info(_data("5% (1.7 laps)", "0.86")).energy_per_lap == pytest.approx(0.05 / 1.7)


def test_fuel_ratio_from_its_text() -> None:
    """The fuel ratio is the number shown."""
    assert garage_info(_data("100% (36.2 laps)", "0.86")).fuel_ratio == 0.86


def test_unmatched_text_gives_no_figures() -> None:
    """Text that is not a number, zero laps or a missing setting gives None for that figure."""
    assert garage_info(_data("N/A", "N/A")) == GarageInfo(None, None)
    assert garage_info(_data("0% (0.0 laps)", "0.5")) == GarageInfo(0.5, None)
    assert garage_info({}) == GarageInfo(None, None)


def test_none_input_gives_none() -> None:
    """A null body gives no garage info."""
    assert garage_info(None) is None


def test_merge_keeps_the_last_good_value_per_figure() -> None:
    """A read without one of the figures, without both or without a reply does not wipe what was read before."""
    good = GarageInfo(0.69, 0.044)
    assert merge_garage(None, None) is None
    assert merge_garage(None, good) == good
    assert merge_garage(good, None) == good
    assert merge_garage(good, GarageInfo(None, None)) == good
    assert merge_garage(good, GarageInfo(0.7, None)) == GarageInfo(0.7, 0.044)
    assert merge_garage(good, GarageInfo(None, 0.05)) == GarageInfo(0.69, 0.05)
