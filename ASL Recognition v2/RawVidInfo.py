import numpy as np


class RawVidInfo:
    def __init__(self, right_hand, left_hand, face, scaling_dist):
        """
        :param right_hand: positions for right hand
        :param left_hand: positions for left hand
        :param face: positions for the face
        :param scaling_dist: scaling dist

        this class holds the raw information for each sign
        """
        self.right_hand = np.array([np.array([pos.x, pos.y]) for pos in right_hand])
        self.left_hand = np.array([np.array([pos.x, pos.y]) for pos in left_hand])
        self.face = np.array([face[0].x, face[0].y])
        self.scaling_dist = scaling_dist
