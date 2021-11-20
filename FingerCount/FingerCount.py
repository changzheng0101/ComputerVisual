import cv2.cv2 as cv2
import time
import os
import HandTrackingMoudle as HTM

#######################
CamWidth, CamHeight = 640, 480
#######################

cap = cv2.VideoCapture(0)
cap.set(3, CamWidth)
cap.set(4, CamHeight)
pTime = 0
detector = HTM.handDetector(min_detection_confidence=0.75)
fingerImgPath = "../data/finger"
FingerImgList = []
for imgPath in os.listdir(fingerImgPath):
    img = cv2.imread(os.path.join(fingerImgPath, imgPath))
    FingerImgList.append(img)
fingerTipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    landmarkList = detector.drawHandsAndGetPosition(img)
    if len(landmarkList) > 0:
        fingers = []
        if landmarkList[fingerTipIds[0]][1] > landmarkList[fingerTipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for fingerIndex in range(1, 5):
            if landmarkList[fingerTipIds[fingerIndex]][2] < landmarkList[fingerTipIds[fingerIndex] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        h, w, c = FingerImgList[totalFingers].shape
        img[20:20 + h, 20:20 + w] = FingerImgList[totalFingers]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps:{int(fps)}", (500, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
