import cv2.cv2 as cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 前置和后置时间
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)
    # 进行画线
    if results.multi_hand_landmarks:
        # handLandmark 具体的手的坐标
        for handLandmark in results.multi_hand_landmarks:
            h, w, c = img.shape
            # print(handLandmark.landmark)
            for id, lm in enumerate(handLandmark.landmark):
                hand_x, hand_y = int(w * lm.x), int(h * lm.y)
                if id == 0:
                    cv2.circle(img, (hand_x, hand_y), 15, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
