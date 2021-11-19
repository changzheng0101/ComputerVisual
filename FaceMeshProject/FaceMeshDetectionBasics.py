import cv2.cv2 as cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("../data/1.mp4")
pTime = 0

mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandMark in results.multi_face_landmarks:
            # 后面两个参数 一个决定画的点 一个决定连的线
            mpDraw.draw_landmarks(img, faceLandMark, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            h, w, c = img.shape
            for id, lm in enumerate(faceLandMark.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps:{int(fps)}", (70, 70), 5, cv2.FONT_HERSHEY_PLAIN, (255, 255, 0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
