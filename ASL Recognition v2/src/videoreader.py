import cv2
import mediapipe as mp
import math
from google.protobuf.json_format import MessageToDict
import numpy as np


class VideoReader:
    def __init__(self, option, video_path=None):
        """
        :param option: value of 0 or 1 where 0 represents livestream and 1 represents video upload
        :param video_path: path to the video if option is video upload.  defaulted to None
        """
        if option == 0:
            self.cap = cv2.VideoCapture(0)
        elif option == 1:
            self.cap = cv2.VideoCapture(video_path)
        else:
            raise "Invalid value in parameter option"

        self.hand_detector = mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.backup_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def get_hand_location(handedness, landmarks):
        """
        :param handedness: handedness for a singular hand
        :param landmarks:  landmarks for a singular hand
        :return: the x, y coordinates of the middle finger dip (taken as location of hand due to central location)
        """
        # the x, y coordinates of the middle finger dip (take as location of hand)
        # left hand is index 0 and right hand is index 1

        handedness_dict = MessageToDict(handedness)['classification'][0]
        handedness_dict.pop('score')

        landmarks_dict = MessageToDict(landmarks)['landmark'][9]
        landmarks_dict.pop('z')

        return handedness_dict, landmarks_dict

    @staticmethod
    def get_face_location(face) -> np.array:
        box = face.detections[0].location_data.relative_bounding_box
        return np.array([
            round(box.xmin + box.width / 2, 5),
            round(box.ymin + box.height / 2, 5)
        ])

    @staticmethod
    def get_scaling_dist(face) -> float:
        box = face.detections[0].location_data.relative_bounding_box
        return round(math.sqrt(box.width * box.width + box.height * box.height), 5)

    def get_backup(self):
        frame_num = 0
        self.cap.set(1, frame_num)
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            face_result = self.face_detector.process(frame)
            if face_result.detections is None:
                face_result = self.backup_face_detector.process(frame)

            if face_result.detections is not None:
                # get face location and scaling dist (only at first frame face appears)
                face_location = self.get_face_location(face_result)
                scaling_dist = self.get_scaling_dist(face_result)

                return face_location, scaling_dist

    def read(self, start_frame=0, end_frame=math.inf, show=False) -> (np.array, float, np.array):
        """
        :param start_frame: frame number of the start of the portion of video, inclusive
        :param end_frame: frame number of the end of the portion of video, inclusive
        :param show: whether to display the video or not
        :return: location of face, scaling distance, sign datapoints
        """
        # data variables
        sign = []
        face_location = None
        scaling_dist = None

        frame_num = start_frame
        self.cap.set(1, frame_num)
        while frame_num <= end_frame:
            success, frame = self.cap.read()
            if not success:
                frame_num += 1
                continue

            frame_num += 1

            face_result = self.face_detector.process(frame)
            if face_result is None:
                face_result = self.backup_face_detector.process(frame)

            if face_result.detections is not None:
                if show:
                    self.mp_drawing.draw_detection(frame, face_result.detections[0])
                # get face location and scaling dist (only at first frame face appears)
                if face_location is None:
                    face_location = self.get_face_location(face_result)
                    scaling_dist = self.get_scaling_dist(face_result)

            # get hand results
            hand_results = self.hand_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                data_point = [[0, 0], [0, 0]]
                changed = [False, False]
                for hand_handedness, landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
                    # display landmarks on the screen
                    if show:
                        self.mp_drawing.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    handedness_dict, landmarks_dict = self.get_hand_location(hand_handedness, landmarks)

                    if changed[handedness_dict['index']]:
                        pass

                    rounded_x, rounded_y = round(landmarks_dict['x'], 5), round(landmarks_dict['y'], 5)
                    data_point[handedness_dict['index']] = np.array([rounded_x, rounded_y])
                    changed[handedness_dict['index']] = True

                sign.append(np.array(data_point))

            if show:
                cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return face_location, scaling_dist, np.array(sign)
