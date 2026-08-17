import numpy as np

from f1_pit_strategy.active_learning import uncertainty


def test_uncertainty_strategies_rank_ambiguous_predictions_higher() -> None:
    probabilities = np.array([[.9, .1], [.51, .49]])
    for strategy in ("least_confidence", "margin", "entropy"):
        assert uncertainty(probabilities, strategy)[1] > uncertainty(probabilities, strategy)[0]
