import cv2
import numpy as np

class CameraModule():
    def __init__(self, index):
        self.cam = cv2.VideoCapture(index)
    
    def camera_test(self):
        winName = 'image'
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        while(1):
            ret, frame = self.cam.read()
            x_bias = 0
            y_bias = -10
            x = 290+x_bias
            y = 270+y_bias
            w = 10
            h = 10
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if not ret:
                break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
            cv2.imshow(winName , gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def get_camera_frame(self, file_name):
        s, im = self.cam.read()
        if file_name:
            return cv2.imwrite(file_name, im)
        return im
        
    def object_track(self):
        object_tracker = {
            'csrt': cv2.TrackerCSRT_create,
            'kcf': cv2.TrackerKCF_create,
            'boosting': cv2.legacy.TrackerBoosting_create,
            'mil': cv2.TrackerMIL_create,
            'tld': cv2.legacy.TrackerTLD_create,
            'medianflow': cv2.legacy.TrackerMedianFlow_create,
            'mosse': cv2.legacy.TrackerMOSSE_create
        }
        trackers = cv2.legacy.MultiTracker_create()
        while True:
            ret, frame = self.cam.read()
            (h, w) = frame.shape[:2]
            width = 600
            r = width/float(w)
            dim = (width, int(h*r))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(100)&0xFF
            if key == ord('s'):
                box = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
                tracker = cv2.legacy.TrackerCSRT_create()
                trackers.add(tracker, frame, box)
        
if __name__ == '__main__':
    cam = CameraModule(1)
    cam.camera_test()