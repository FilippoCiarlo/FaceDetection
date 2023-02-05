import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectCon=0.5, modelSelection=0):
        self.minDetectCon = minDetectCon
        self.modelSelection = modelSelection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectCon,self.modelSelection)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.advancedDraw(img,bbox)
                    #cv2.rectangle(img, bbox, (0,255,0), 2)
                    cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img, bboxs


    def advancedDraw(self, img, bbox, l=30, rt=1, t=2):
        x, y, w, h = bbox   # (x,y) = origin
        x1, y1 = x+w, y+h   # (x1,y1) = bottom right corner point
        r, g, b = 80, 255, 222
        # Rectangle
        cv2.rectangle(img, bbox, (r,g,b), rt)
        # Top-left corner (x,y)
        cv2.line(img, (x,y), (x+l,y), (r,g,b), t)
        cv2.line(img, (x,y), (x,y+l), (r,g,b), t)
        # Top-right corner (x1,y)
        cv2.line(img, (x1,y), (x1-l,y), (r,g,b), t)
        cv2.line(img, (x1,y), (x1,y+l), (r,g,b), t)
        # Bottom-left corner (x,y1)
        cv2.line(img, (x,y1), (x+l,y1), (r,g,b), t)
        cv2.line(img, (x,y1), (x,y1-l), (r,g,b), t)
        # Bottom-right corner (x1,y1)
        cv2.line(img, (x1,y1), (x1-l,y1), (r,g,b), t)
        cv2.line(img, (x1,y1), (x1,y1-l), (r,g,b), t)

        return img


def main():
    cap = cv2.VideoCapture("Videos/1.mp4")
    #cap = cv2.VideoCapture(0)

    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img, draw=True)
        #print(bboxs)

        # calculate FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # visualize output
        cv2.putText(img, f"FPS:{str(int(fps))}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (80, 255, 222), 2)
        cv2.imshow("FaceDetection",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()