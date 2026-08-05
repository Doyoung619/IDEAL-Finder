from experiments.design import assigned_order, balanced_latin_square


def test_balanced_latin_square_contains_each_condition_once():
    rows = balanced_latin_square([2, 4, 8, 16])
    assert len(rows) == 4
    assert all(sorted(row) == [2, 4, 8, 16] for row in rows)


def test_assignment_is_stable():
    first = assigned_order([2, 4, 8, 16], "ITD-ABC", 99)
    second = assigned_order([2, 4, 8, 16], "ITD-ABC", 99)
    assert first == second

