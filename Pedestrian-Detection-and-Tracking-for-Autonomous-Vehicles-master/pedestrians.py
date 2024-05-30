import sys
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import matplotlib.pyplot as plt



class pedestrianTracking():
    
    def __init__(self, id, frame, bound_box):

        self.id = int(id)
        x,y,w,h = bound_box
        self.tracking_window = bound_box
        
        self.regionInterest = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([self.regionInterest], [0], None, [16], [0, 180])

        self.norm_hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        # Kalman filter
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        self.measurement = np.array((2,1), np.float32) 
        self.prediction = np.zeros((2,1), np.float32)
        self.term_criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.center = None
        self.update_predict(frame)

    def update_predict(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        back_project = cv2.calcBackProject([hsv],[0], self.norm_hist,[0,180],1)

        ret, self.tracking_window = cv2.meanShift(back_project, self.tracking_window, self.term_criteria)

        x,y,w,h = self.tracking_window

        self.center = self.findCenter([[x,y],[x+w, y],[x,y+h],[x+w, y+h]])  
        
        
        self.kalman.correct(self.center)
        
        
        predicted = self.kalman.predict()
        cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 4, (255, 0, 0), -1)

        
    def findCenter(self,points):
        
        self.x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
        self.y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
        return np.array([np.float32(self.x), np.float32(self.y)], np.float32)
    


def main():
    # read the video
    readVideo = cv2.VideoCapture("london_bus.mp4")


    cv2.namedWindow("Pedestrian Detection")
    
    
    detectedPedestrians = {}
    firstFrame = True
    frames = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 240.0, (640,480))
    pauseVideo = False
    while True:

        if pauseVideo ==False:
            flagCaptured, frame = readVideo.read()
        if (flagCaptured is False):
            print("could not get frame")
            break

        #HOG descriptor 
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                                padding=(8, 8), scale=1.05,hitThreshold=0.22)       
         
        rectBoxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        suppressedRectBoxes = non_max_suppression(rectBoxes, probs=None, overlapThresh=0.95)
        counter = 0

        #final bounding boxes
        for (xA, yA, xB, yB) in suppressedRectBoxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
            if firstFrame is True:
                detectedPedestrians[counter] = pedestrianTracking(counter, frame,
                                                                  (xA,yA,abs(xB-xA),abs(yB-yA)))
                counter += 1
        
        for key, value in detectedPedestrians.items():
            value.update_predict(frame)

        firstFrame = False
        frames += 1

        cv2.imshow("Pedestrian Detection", frame)
        out.write(frame)
        
        # press ESC to close video window
        if (cv2.waitKey(10) & 0xFF) == 27:
            cv2.destroyWindow("Pedestrian Detection")
            break
            
        # press spacebar to pause video
        if (cv2.waitKey(10) & 0xFF) == 32:
            print('Video paused')
            pauseVideo = True
            
        # press enter to resume
        if (cv2.waitKey(10) & 0xFF) == 13:
            print('Video resumed')
            pauseVideo = False
            
            
            
    out.release()
    readVideo.release()


if __name__ == "__main__":
    main()







