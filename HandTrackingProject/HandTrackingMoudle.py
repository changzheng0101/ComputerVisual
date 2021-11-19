import cv2.cv2 as cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode,
                                        max_num_hands,
                                        model_complexity,
                                        min_detection_confidence,
                                        min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, is_draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            # handLandmark 具体的手的坐标
            for handLandmark in results.multi_hand_landmarks:
                if is_draw:
                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)

    def findPosition(self, img, hand_index=0):
        #########多进行一次检测会导致性能下降很多
        lmList = []  # 存放landmark
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > hand_index:
                # 获取某只手的所有坐标
                handLandmark = results.multi_hand_landmarks[hand_index]
                h, w, c = img.shape
                for id, lm in enumerate(handLandmark.landmark):
                    hand_x, hand_y = int(w * lm.x), int(h * lm.y)
                    lmList.append([id, hand_x, hand_y])
        return lmList

    def drawHandsAndGetPosition(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        lmList = []

        if results.multi_hand_landmarks:
            # handLandmark 具体的手的坐标
            handLandmark = results.multi_hand_landmarks[0]
            h, w, c = img.shape
            for id, lm in enumerate(handLandmark.landmark):
                hand_x, hand_y = int(w * lm.x), int(h * lm.y)
                lmList.append([id, hand_x, hand_y])
            for handLandmark in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    # 前置和后置时间
    pTime = 0
    cTime = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        detector.findHands(img)

        cv2.putText(img, "fps:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255))

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
