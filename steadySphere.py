import cv2 as cv
import numpy as np

def getRangeHSV(self,BGR_color):
        BGR_color=np.uint8([[BGR_color]])
        HSV_color=cv.cvtColor(BGR_color,cv.COLOR_BGR2HSV)
