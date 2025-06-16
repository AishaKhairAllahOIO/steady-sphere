import cv2 as cv
import numpy as np

ballCenter_X=0
ballCenter_Y=0

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
    frame=cv.flip(frame,1)
    HSV_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    mask=cv.inRange(HSV_frame,lower_yellow,upper_yellow)
    blurred=cv.GaussianBlur(mask,(9,9),0)
    kernel=np.ones((5, 5),np.uint8)
    mask_clean=cv.morphologyEx(blurred,cv.MORPH_OPEN,kernel)
    mask_clean=cv.morphologyEx(mask_clean,cv.MORPH_CLOSE,kernel)
    contours, _=cv.findContours(mask_clean,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour= max(contours,key=cv.contourArea)
        ((ballCenter_X,ballCenter_Y),radius)=cv.minEnclosingCircle(largest_contour)
        M=cv.moments(largest_contour)
        if M["m00"]>0:
            ballCenter_X =int(M["m10"]/M["m00"])
            ballCenter_Y=int(M["m01"]/M["m00"])




    cv.imshow("Ball Tracking",frame)
    if not ret:
        break
    if cv.waitKey(1)&0xFF==ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()