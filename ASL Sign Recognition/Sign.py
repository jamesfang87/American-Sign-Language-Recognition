import numpy as np
from tslearn.metrics import dtw as ts_dtw
from SignInfo import SignInfo
from scipy.spatial.distance import euclidean


class Sign:
    def __init__(self, gloss: str, num_hands: int, examples: list[SignInfo]):
        self.gloss: str = gloss
        self.num_hands: int = num_hands  # both this class and SignInfo contain num_hands. this is done for convenience
        self.examples: list[SignInfo] = examples
        self.confidence_score = None

    @staticmethod
    def __distance(s1: np.ndarray, s2: np.ndarray) -> float:
        base_feature_weights = [2, 2, 1, 1]
        all_feature_weights = [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1]

        selected = base_feature_weights if len(s1) == 4 else all_feature_weights
        euclidean_dist = euclidean(s1, s2, selected)
        return euclidean_dist

    def calc_distance(self, other: SignInfo) -> float:
        """
        "distance" between this sign and another calculated with DTW

        call only when the number of hands is confirmed to be the same
        """

        confidence_scores = []
        for example in self.examples:
            # costs for position of dominant and non-dominant hands
            cost = ts_dtw(example.all_features,
                          other.all_features,)

            confidence_scores.append(cost)

        # the best score is taken as the score for the sign
        return min(confidence_scores)
