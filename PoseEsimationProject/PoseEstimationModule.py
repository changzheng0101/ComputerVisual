import cv2.cv2 as cv2
import mediapipe as mp
import time


class poseDetector:
    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode, model_complexity, smooth_landmarks,
                                     enable_segmentation, smooth_segmentation, min_detection_confidence,
                                     min_tracking_confidence)

    def findPose(self, img, is_draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if is_draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

    def findPosition(self, img):
        lmList = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        h, w, c = img.shape
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(w * lm.x), int(h * lm.y)
            lmList.append([id, cx, cy])
        return lmList


def main():
    cap = cv2.VideoCapture("../data/1.mp4")
    detector = poseDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) > 0:
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (255, 3, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, "fps" + str(int(fps)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
