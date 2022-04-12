import cv2
import numpy as np

class CameraModule():
    def __init__(self, index):
        self.cam = cv2.VideoCapture(index)
    
    def camera_test(self):
        winName = 'image'
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        while(1):
            s, im = self.cam.read()
            im_gray = im[:,:,0]*0.21+im[:,:,1]*0.72+im[:,:,2]*0.07
            cv2.imshow(winName , im_gray/255)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def get_camera_frame(self, file_name):
        s, im = self.cam.read()
        if file_name:
            return cv2.imwrite(file_name, im)
        return im
if __name__ == '__main__':
    cam = CameraModule(1)
    cam.camera_test()