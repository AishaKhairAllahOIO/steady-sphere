import cv2 as cv
import numpy as np

def getRangeHSV(BGR_color):
        BGR_color=np.uint8([[BGR_color]])
        HSV_color=cv.cvtColor(BGR_color,cv.COLOR_BGR2HSV)

        lower_color=HSV_color[0][0][0]-10,100,100
        upper_color=HSV_color[0][0][0]+10,255,255

        lower_color=np.array(lower_color,np.uint8)
        upper_color=np.array(upper_color,np.uint8)

        return lower_color,upper_color


yellow=[0,255,255]
green=[0,255,0]

lower_yellow,upper_yellow=getRangeHSV(yellow)
lower_green, upper_green =getRangeHSV(green)

print("Yellow color HSV range:")
print("Lower bound:", lower_yellow)
print("Upper bound:", upper_yellow)

print("\nGreen color HSV range:")
print("Lower bound:", lower_green)
print("Upper bound:", upper_green)

videoCapture=cv.VideoCapture(0)
cv.namedWindow("Ball Tracking")
cv.resizeWindow("Ball Tracking",800,600)

while True:
    ret,frame=videoCapture.read()
    
    cv.imshow("Ball Tracking",frame)
    if not ret:
        break
    if cv.waitKey(1)&0xFF==ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()