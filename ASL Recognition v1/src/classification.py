from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math
from sign import Sign


def classify(testing_data: list[Sign], examples: list[Sign]):
    for unknown_sign in testing_data:
        prediction, min_cost = -1, math.inf
        for i, example in enumerate(examples):
            if example.handedness == unknown_sign.handedness:
                left_hand_cost, _ = fastdtw(unknown_sign.left_hand, example.left_hand, dist=euclidean)
                right_hand_cost, _ = fastdtw(unknown_sign.right_hand, example.right_hand, dist=euclidean)
                right_hand_motion_cost, _ = fastdtw(unknown_sign.right_hand_motion, example.right_hand_motion,
                                                    dist=euclidean)
                left_hand_motion_cost, _ = fastdtw(unknown_sign.left_hand_motion, example.left_hand_motion,
                                                   dist=euclidean)
                hand_dist_cost, _ = fastdtw(unknown_sign.hand_dist, example.hand_dist, dist=euclidean)
                hand_dist_change_cost, _ = fastdtw(unknown_sign.hand_dist_change, example.hand_dist_change,
                                                   dist=euclidean)
                total_cost = (
                        left_hand_cost +
                        right_hand_cost +
                        right_hand_motion_cost +
                        left_hand_motion_cost +
                        hand_dist_cost +
                        hand_dist_change_cost
                )
                if total_cost < min_cost:
                    prediction = examples[i].gloss
                    min_cost = total_cost

        print(prediction)
