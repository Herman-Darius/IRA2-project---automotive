import sys
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils

class CarTracking:
    def __init__(self, id, frame, bound_box):
        self.id = int(id)
        x, y, w, h = bound_box
        self.tracking_window = bound_box
        self.regionInterest = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([self.regionInterest], [0], None, [16], [0, 180])
        self.norm_hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update_predict(frame)

    def update_predict(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.norm_hist, [0, 180], 1)
        ret, self.tracking_window = cv2.meanShift(back_project, self.tracking_window, self.term_criteria)
        x, y, w, h = self.tracking_window
        self.center = self.findCenter([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.kalman.correct(self.center)
        predicted = self.kalman.predict()
        cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 4, (255, 0, 0), -1)

    def findCenter(self, points):
        self.x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
        self.y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
        return np.array([np.float32(self.x), np.float32(self.y)], np.float32)


def main():
    #video ---->
    video_path = "london_bus.mp4" 

    # <----
    readVideo = cv2.VideoCapture(video_path)

    if not readVideo.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    cv2.namedWindow("Car Detection")

    detectedCars = {}
    firstFrame = True
    frames = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    pauseVideo = False

    car_cascade = cv2.CascadeClassifier('cars.xml')

    while True:
        if not pauseVideo:
            flagCaptured, frame = readVideo.read()
            if not flagCaptured:
                print("Error: Could not read frame")
                break

        # Convert to grayscale for car detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(40, 30))

        rectBoxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
        suppressedRectBoxes = non_max_suppression(rectBoxes, probs=None, overlapThresh=0.65)
        counter = 0
        for (xA, yA, xB, yB) in suppressedRectBoxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            if firstFrame:
                detectedCars[counter] = CarTracking(counter, frame, (xA, yA, abs(xB - xA), abs(yB - yA)))
                counter += 1

        for key, value in detectedCars.items():
            value.update_predict(frame)

        firstFrame = False
        frames += 1

        cv2.imshow("Car Detection", frame)
        out.write(frame)

        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC key
            cv2.destroyWindow("Car Detection")
            break
        elif key == 32:  # Spacebar to pause
            print('Video paused')
            pauseVideo = True
        elif key == 13:  # Enter to resume
            print('Video resumed')
            pauseVideo = False

    out.release()
    readVideo.release()


if __name__ == "__main__":
    main()
