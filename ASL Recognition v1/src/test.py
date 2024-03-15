import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import math

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

cap = cv2.VideoCapture(0)
frame_num = 44
cap.set(1, frame_num)
end = math.inf
while cap.isOpened():
    if end is not None and end <= frame_num:
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

    cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.set(1, 50)
end = 59
while cap.isOpened():
    if end is not None and end <= frame_num:
        break

    success, frame = cap.read()
    if not success:
        print("asdf")
        break

    frame_num += 1
    frame = cv2.flip(frame, 1)

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

    cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame_num = 50
cap.set(1, frame_num)
end = 59
while cap.isOpened():
    if end is not None and end <= frame_num:
        break

    success, frame = cap.read()
    if not success:
        print('asdf')
        break

    frame_num += 1
    frame = cv2.flip(frame, 1)

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

    cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


