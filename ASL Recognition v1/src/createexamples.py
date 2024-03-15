from videoreader import VideoReader
from sign import Sign
import numpy as np
import pandas as pd
from copy import deepcopy


def create_examples(path):
    signs = pd.read_csv(f'../{path}/list/{path}_annotate.csv', nrows=20)
    signs.ffill(inplace=True)

    backup_face_location, backup_scaling_dist = np.array([0.5128158926963806, 0.34264886379241943]), 0.2649799037231937

    examples = []
    prev_vid = 'filler'
    vid = None  # vid reader, will get initialized first iteration
    for _, video, gloss, start_frame, end_frame in np.array(signs):
        if video != prev_vid:
            vid = VideoReader(1, f'../{path}/camera 1/{video}.mov')
            backup_face_location, backup_scaling_dist = vid.get_backup()
            prev_vid = deepcopy(video)

        face_location, scaling_dist, hand_locations = vid.read(start_frame, end_frame)

        if face_location is None:
            face_location = backup_face_location

        if scaling_dist is None:
            scaling_dist = backup_scaling_dist

        examples.append(Sign(face_location, scaling_dist, hand_locations, 2, gloss))

    return examples
