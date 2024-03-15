import numpy as np
from copy import deepcopy


class Sign:
    def __init__(self, face_location: np.array, scaling_dist: float, sign: np.array, handedness, gloss=None):
        self.gloss = gloss  # meaning of sign, defaulted to None in cases where it is not known
        self.handedness = handedness  # number of hands

        sign -= face_location
        sign /= scaling_dist

        # len(sign) x 2 matrix with location of hands
        self.left_hand = deepcopy(sign[:, 0])  # motion of left hand
        self.right_hand = deepcopy(sign[:, 1])  # motion of right hand

        self.left_hand_motion = []  # unit vectors representing motion of left hand
        self.right_hand_motion = []  # unit vectors representing motion of right hand
        for i in range(len(sign) - 1):
            left_hand_norm = np.linalg.norm(self.left_hand[i + 1] - self.left_hand[i])
            right_hand_norm = np.linalg.norm(self.right_hand[i + 1] - self.right_hand[i])

            if left_hand_norm == 0.0:
                left_hand_normalized = np.zeros_like([0, 0])
            else:
                left_hand_normalized = ((self.left_hand[i + 1] - self.left_hand[i]) /
                                        left_hand_norm)
            if right_hand_norm == 0.0:
                right_hand_normalized = np.zeros_like([0, 0])
            else:
                right_hand_normalized = ((self.right_hand[i + 1] - self.right_hand[i]) /
                                         right_hand_norm)

            self.left_hand_motion.append(left_hand_normalized)
            self.right_hand_motion.append(right_hand_normalized)
        self.left_hand_motion = np.array(self.left_hand_motion)
        self.right_hand_motion = np.array(self.right_hand_motion)

        # distance between the two hands
        self.hand_dist = deepcopy(self.left_hand - self.right_hand)

        # unit vectors representing the change of the distance between both hands
        self.hand_dist_change = []
        for i in range(len(sign) - 1):
            hand_dist_norm = np.linalg.norm(self.hand_dist[i + 1] - self.hand_dist[i])
            if hand_dist_norm == 0.0:
                temp = np.array([0, 0])
            else:
                temp = np.array((self.hand_dist[i + 1] - self.hand_dist[i]) /
                                hand_dist_norm)
            self.hand_dist_change.append(temp)

        self.hand_dist_change = np.array(self.hand_dist_change)
