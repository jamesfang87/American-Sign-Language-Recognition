from videoreader import VideoReader
from sign import Sign
import numpy as np
import pandas as pd


def annotate(path, n):
    """
    :param path: path to the ASL session folder
    :param n: creates examples with the first n signs
    """
    # files inside annotations to write in:
    face_location_writer = open(f'../{path}/annotations/face_location.csv', 'w')
    scaling_dist_writer = open(f'../{path}/annotations/scaling_dist.csv', 'w')
    hand_location_writer = open(f'../{path}/annotations/hands.csv', 'w')
    gloss_writer = open(f'../{path}/annotations/glosses.csv', 'w')

    face_location_writer.write('x,y\n')
    scaling_dist_writer.write('scaling_dist\n')
    hand_location_writer.write(f'{n}\n')
    gloss_writer.write('gloss\n')

    # get signs from csv file in list folder in ASL session folder
    signs: pd.DataFrame = pd.read_csv(f'../{path}/list/{path}_annotate.csv', nrows=n)
    # forward fill missing scene-camera data
    signs.ffill(inplace=True)

    prev_vid = None
    vid = None  # vid reader, will get initialized first iteration

    backup_face_location, backup_scaling_dist = None, None  # will get initialized first iteration

    for _, video, gloss, start_frame, end_frame in np.array(signs):
        # initialize vid reader for first iteration or for new video
        if prev_vid is None or video != prev_vid:
            vid = VideoReader(1, f'../{path}/camera 1/{video}.mov')
            # get backup face location and scaling dist from first frames in new video
            backup_face_location, backup_scaling_dist = vid.get_backup()
            # update previous vid
            prev_vid = video

        # get face location, scaling dist and hand locations
        face_location, scaling_dist, hand_locations = vid.read(start_frame, end_frame)

        # use backups if needed
        if face_location is None:
            face_location = backup_face_location

        if scaling_dist is None:
            scaling_dist = backup_scaling_dist

        # record face location, scaling dist, and gloss
        face_location_writer.write(f'{face_location[0]},{face_location[1]}\n')
        scaling_dist_writer.write(f'{scaling_dist}\n')
        gloss_writer.write(f'{gloss}\n')

        # hand location is written as left-x, left-y, right-x, right-y in same line
        hand_location_writer.write(str(len(hand_locations)) + '\n')
        for hand_location in hand_locations:
            hand_location_writer.write(f'{hand_location[0][0]},'
                                       f'{hand_location[0][1]},'
                                       f'{hand_location[1][0]},'
                                       f'{hand_location[1][1]}\n')

    face_location_writer.close()
    scaling_dist_writer.close()
    hand_location_writer.close()
    gloss_writer.close()


def create_examples(path) -> list[Sign]:
    """
    :param path: path to the ASL session folder
    :return: a list of Sign objects
    """

    face_locations = np.array(pd.read_csv(f'../{path}/annotations/face_location.csv'))
    scaling_dists = np.array(pd.read_csv(f'../{path}/annotations/scaling_dist.csv'), dtype=float)
    glosses = np.array(pd.read_csv(f'../{path}/annotations/glosses.csv')['gloss'], dtype=str)

    hand_location_reader = open(f'../{path}/annotations/hands.csv', 'r')
    n = int(hand_location_reader.readline().strip('\n'))

    examples: list[Sign] = []
    for i in range(n):
        sign_len = int(hand_location_reader.readline().strip('\n'))
        sign = []
        for _ in range(sign_len):
            line = list(map(float, hand_location_reader.readline().strip('\n').split(',')))
            sign.append(np.array([np.array([line[0], line[1]]), np.array([line[2], line[3]])]))

        examples.append(Sign(face_locations[i], float(scaling_dists[i]), sign, 2, glosses[i]))

    return examples
