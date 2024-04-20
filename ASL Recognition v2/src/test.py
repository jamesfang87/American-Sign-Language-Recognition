import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import math
import time

mp_drawing = mp.solutions.drawing_utils

# initialize hand landmark detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4)

# initialize face detection model
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture('../ASL_2008_01_11/camera 1/scene21-camera1.mov')
frame_num = 696
cap.set(cv2.CAP_PROP_FPS, 60)
print(cap.get(cv2.CAP_PROP_FPS))
cap.set(1, frame_num)
cap.set(0, frame_num * 1000 / 60)
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print(int(cap.get(cv2.CAP_PROP_FPS)))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = 60
print(frame_count / 60)
cap.set(0, 1000 * (696 / frame_count) * (frame_count / fps))
print((696 / (int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) * (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 60)))

while cap.isOpened():
    print(frame_num)
    if frame_num > 730:
        break

    success, frame = cap.read()
    if not success:
        break

    frame_num += 1

    # get face location (only at first frame)
    frame_height, frame_width, _ = frame.shape

    face = face_detector.process(frame)
    if face.detections is not None:
        box = face.detections[0].location_data.relative_bounding_box
        width = box.width / 2
        height = box.height / 2
        face_location = (
            box.xmin + width,
            box.ymin + height,
        )
        scaling_dist = (width + height) / 2

    # get hand results
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_handedness and results.multi_hand_landmarks:
        for hand_handedness, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            handedness_dict = MessageToDict(hand_handedness)['classification'][0]
            handedness_dict.pop('score')

            landmarks_dict = MessageToDict(landmarks)['landmark'][9]
            landmarks_dict.pop('z')
            print(handedness_dict, landmarks_dict)



    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        temp = [[0, 0], [0, 0]]
        for i in range(len(results.multi_hand_landmarks)):
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Pose', frame)
    time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
