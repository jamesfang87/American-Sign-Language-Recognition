import cv2
import mediapipe as mp
import math
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from RawVidInfo import RawVidInfo


class VideoReader:
    def __init__(self, option, video_path=None):
        """
        :param option: value of 0 or 1 where 0 represents livestream and 1 represents video upload
        :param video_path: path to the video if option is video upload.  defaulted to None
        """
        if option == 0:
            self.cap = cv2.VideoCapture(0)  # capture from livestream
        elif option == 1:
            self.cap = cv2.VideoCapture(video_path)  # capture from video
        else:
            raise "Invalid value in parameter option"  # raise error

        self.mp_pose = mp.solutions.pose
        self.pose_detector = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

    def read(self, start_frame=0, end_frame=math.inf, show=False) -> (np.array, np.array, np.array, float):
        """
        :param start_frame: frame number of the start of the portion of video, inclusive
        :param end_frame: frame number of the end of the portion of video, inclusive
        :param show: whether to display the video or not
        :return: location of face, scaling distance, hand locations
        """
        # data variables
        right_hand_positions = []
        left_hand_positions = []
        face_positions = []
        scaling_dist = 0

        frame_num = start_frame
        self.cap.set(1, frame_num)
        while self.cap.isOpened() and frame_num <= end_frame:
            # increment frame number
            frame_num += 1

            # read frame
            successful, frame = self.cap.read()
            if not successful:
                continue

            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process the frame for pose detection
            pose_result = self.pose_detector.process(frame)
            face_result = self.face_detector.process(frame)

            right_hand_positions.append(pose_result.pose_landmarks.landmark[PoseLandmark.RIGHT_INDEX])
            left_hand_positions.append(pose_result.pose_landmarks.landmark[PoseLandmark.LEFT_INDEX])
            face_positions.append(pose_result.pose_landmarks.landmark[PoseLandmark.NOSE])

            frame_height, frame_width, _ = frame.shape
            if show:
                # convert back to BGR for display
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # draw skeleton on the frame
                if pose_result:
                    self.mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # draw face bounding box
                if face_result.detections:
                    face = face_result.detections[0]  # there should only be one face
                    face_rect = np.multiply(
                        [
                            face.location_data.relative_bounding_box.xmin,
                            face.location_data.relative_bounding_box.ymin,
                            face.location_data.relative_bounding_box.width,
                            face.location_data.relative_bounding_box.height,
                        ],
                        [
                            frame_width,
                            frame_height,
                            frame_width,
                            frame_height
                        ]
                    ).astype(int)

                    width = face.location_data.relative_bounding_box.width
                    height = face.location_data.relative_bounding_box.height
                    scaling_dist = (width + height) / 2

                    cv2.rectangle(frame, face_rect, color=(255, 255, 255), thickness=2)

                # display the frame converted back to BGR
                cv2.imshow('', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        if len(face_positions) == 0:
            print("uh oh")

        return RawVidInfo(np.array(right_hand_positions),
                          np.array(left_hand_positions),
                          np.array(face_positions),
                          scaling_dist)
