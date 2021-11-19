import cv2.cv2 as cv2
import mediapipe as mp
import time


class FaceMeshDetection:
    def __init__(self, static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(static_image_mode, max_num_faces, refine_landmarks,
                                                 min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)

    def findFaceMesh(self, img, is_draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.FaceMesh.process(imgRGB)
        faceMeshList = []
        if results.multi_face_landmarks:
            for faceLandMark in results.multi_face_landmarks:
                # 后面两个参数 一个决定画的点 一个决定连的线
                if is_draw:
                    self.mpDraw.draw_landmarks(img, faceLandMark, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                h, w, c = img.shape
                face = []
                for id, lm in enumerate(faceLandMark.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append([id, cx, cy])
                faceMeshList.append(face)
        #首先里面包含几张脸 其次每张脸中拥有对应的点 格式为id,x,y
        return faceMeshList


def main():
    cap = cv2.VideoCapture("../data/1.mp4")
    detector = FaceMeshDetection()
    pTime = 0
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"fps:{int(fps)}", (70, 70), 5, cv2.FONT_HERSHEY_PLAIN, (255, 255, 0))
        detector.findFaceMesh(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
