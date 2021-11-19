import cv2.cv2 as cv2
import time
import numpy as np
import HandTrackingMoudle as HTM
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#######################
CamWidth, CamHeight = 720, 480
#######################

cap = cv2.VideoCapture(0)
cap.set(3, CamWidth)
cap.set(4, CamHeight)
pTime = 0
detector = HTM.handDetector(max_num_hands=1, min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
minVol, maxVol, _ = volume.GetVolumeRange()
vol = np.interp(20, [20, 150], [minVol, maxVol])

while True:
    success, img = cap.read()
    landmarkList = detector.drawHandsAndGetPosition(img)
    if len(landmarkList) > 0:
        # 4 and 8 needed
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        distance = math.hypot(x1 - x2, y1 - y2)
        # distance range 20 150
        # volume range -65.25 0
        vol = np.interp(distance, [20, 150], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        if distance < 20:
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

    # vol bar
    volPos = np.interp(vol, [minVol, maxVol], [400, 150])
    volPercent = np.interp(vol, [minVol, maxVol], [0, 100])
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volPos)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPercent)}%", (50, 440), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps:{int(fps)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
