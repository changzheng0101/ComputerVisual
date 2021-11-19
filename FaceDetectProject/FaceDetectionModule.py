import cv2.cv2 as cv2
import mediapipe as mp
import time


class faceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence, model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, is_draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        face_boxes = []  # 存放所有的脸 xmin ymin width height

        if results.detections:
            for id, detection in enumerate(results.detections):
                boundingBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                boundingBoxInPx = int(boundingBox.xmin * w), int(boundingBox.ymin * h), \
                                  int(boundingBox.width * w), int(boundingBox.height * h)
                face_boxes.append([id, boundingBoxInPx, detection.score])
                if is_draw:
                    cv2.rectangle(img, boundingBoxInPx, (255, 225, 0), 1)
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", (boundingBoxInPx[0], boundingBoxInPx[1] - 20),
                                5, cv2.FONT_HERSHEY_PLAIN, (0, 255, 0))
        return face_boxes


def main():
    cap = cv2.VideoCapture("../data/1.mp4")
    pTime = 0
    detector = faceDetector()
    while True:
        success, img = cap.read()
        detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"fps:{int(fps)}", (70, 70), 5, cv2.FONT_HERSHEY_PLAIN, (255, 255, 0))
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
