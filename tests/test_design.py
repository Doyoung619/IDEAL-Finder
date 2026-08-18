from experiments.design import assigned_order, balanced_latin_square
from app.services.participant_service import experiment_schedule
from app.settings import load_config


def test_balanced_latin_square_contains_each_condition_once():
    rows = balanced_latin_square([2, 4, 8, 16])
    assert len(rows) == 4
    assert all(sorted(row) == [2, 4, 8, 16] for row in rows)


def test_assignment_is_stable():
    first = assigned_order([2, 4, 8, 16], "ITD-ABC", 99)
    second = assigned_order([2, 4, 8, 16], "ITD-ABC", 99)
    assert first == second


def test_six_block_schedule_randomizes_m_and_method_order_reproducibly():
    config = load_config(demo_override=True)
    first = experiment_schedule(config, "ITD-ABC", 99)
    second = experiment_schedule(config, "ITD-ABC", 99)
    assert first == second
    assert len(first) == 6
    assert sorted(item["m"] for item in first) == [2, 2, 4, 4, 8, 8]
    for m_value in (2, 4, 8):
        assert {
            item["strategy"] for item in first if item["m"] == m_value
        } == {"entropy", "rc_mlq"}
    schedules = {
        tuple((item["m"], item["strategy"]) for item in experiment_schedule(config, f"ITD-{index}", 99))
        for index in range(20)
    }
    assert len(schedules) > 4
