import cv2.cv2 as cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()  # 可以进去选择参数

cap = cv2.VideoCapture("../data/1.mp4")

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(w * lm.x), int(h * lm.y)
            #只有id为0才画
            if id == 0:
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "fps" + str(int(fps)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
