import numpy as np

from core.query_novelty import inject_novel_global_candidates


def test_novelty_slots_replace_most_redundant_queries_deterministically():
    history = np.array([[0.0, 0.0], [1.0, 0.0]])
    theta = np.array([[0.01, 0.0], [1.01, 0.0], [0.5, 0.5], [0.5, -0.5]])
    first, metadata = inject_novel_global_candidates(
        theta,
        history,
        np.zeros(2),
        np.eye(2),
        fraction=0.25,
        pool_size=64,
        seed=17,
    )
    second, _ = inject_novel_global_candidates(
        theta,
        history,
        np.zeros(2),
        np.eye(2),
        fraction=0.25,
        pool_size=64,
        seed=17,
    )
    assert metadata["novelty_slots"] == 1
    assert metadata["novelty_replaced_indices"] == [0]
    assert metadata["novelty_min_history_distance"] > 1.0
    assert not np.array_equal(first[0], theta[0])
    assert np.allclose(first[1:], theta[1:])
    assert np.array_equal(first, second)


def test_first_round_preserves_algorithm_queries_without_history():
    theta = np.eye(3, dtype=np.float32)
    final, metadata = inject_novel_global_candidates(
        theta,
        np.empty((0, 3)),
        np.zeros(3),
        np.eye(3),
        fraction=0.25,
        pool_size=32,
        seed=1,
    )
    assert np.array_equal(final, theta)
    assert metadata["novelty_slots"] == 0
