import cv2.cv2 as cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("../data/1.mp4")
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection) #画出框和关键点
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            boundingBox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            boundingBoxInPx = int(boundingBox.xmin * w), int(boundingBox.ymin * h), \
                              int(boundingBox.width * w), int(boundingBox.height * h)
            cv2.rectangle(img, boundingBoxInPx, (255, 225, 0), 1)
            cv2.putText(img, f"{int(detection.score[0] * 100)}%", (boundingBoxInPx[0], boundingBoxInPx[1]-20), 5,
                        cv2.FONT_HERSHEY_PLAIN, (0, 255, 0))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps:{int(fps)}", (70, 70), 5, cv2.FONT_HERSHEY_PLAIN, (255, 255, 0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
