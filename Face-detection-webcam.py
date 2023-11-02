import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

face_recognizer = mp.solutions.face_detection
rectangle = mp.solutions.drawing_utils
face_detector = face_recognizer.FaceDetection()

while cam.isOpened():
    validation, frame = cam.read()
    if not validation:
        pass
    face_list = face_detector.process(frame)

    if face_list.detections:
        for face in face_list.detections:
            rectangle.draw_detection(frame, face)

    cv2.imshow("faces detector in cam", frame)
    if cv2.waitKey(1) == 27:
        break
