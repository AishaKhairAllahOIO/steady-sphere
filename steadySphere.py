import cv2 as cv
import numpy as np
import serial
import time

# arduino=serial.Serial('COM7',9600)
# time.sleep(2)

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
    if not ret:
        break
    frame=cv.flip(frame,1)
    HSV_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    h,s,v=cv.split(HSV_frame)
    v_eq=cv.equalizeHist(v)
    HSV_frame=cv.merge([h,s,v_eq])

    mask=cv.inRange(HSV_frame,lower_yellow,upper_yellow)
    blurred=cv.bilateralFilter(mask,9,75,75)
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    mask_clean=cv.morphologyEx(blurred,cv.MORPH_OPEN,kernel)
    mask_clean=cv.morphologyEx(mask_clean,cv.MORPH_CLOSE,kernel)
    contours, _=cv.findContours(mask_clean,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if contours:
        best_contour=None
        max_score=0
        for contour in contours:
            area=cv.contourArea(contour)
            perimeter=cv.arcLength(contour,True)
            if perimeter==0:
                continue
            circularity=4*np.pi*area/(perimeter**2)
            if circularity>0.7 and area>300: 
                if area>max_score:
                    max_score=area
                    best_contour=contour
        if best_contour is not None:
            ((ballCenter_X,ballCenter_Y),radius)=cv.minEnclosingCircle(best_contour)
            M=cv.moments(best_contour)
            if M["m00"]>0:
                ballCenter_X =int(M["m10"]/M["m00"])
                ballCenter_Y=int(M["m01"]/M["m00"])
                if radius > 10:
                    cv.circle(frame,(int(ballCenter_X),int(ballCenter_Y)),int(radius),(0, 0, 255),4)

                    text_size,_=cv.getTextSize(f"X={int(ballCenter_X)}, Y={int(ballCenter_Y)}",cv.FONT_HERSHEY_SIMPLEX,0.6,2)

                    text_x=ballCenter_X+10
                    text_y=ballCenter_Y-10
                    rect_start=(text_x-5,text_y-text_size[1]-5)
                    rect_end=(text_x+text_size[0]+5,text_y+5)
                    cv.rectangle(frame, rect_start, rect_end, (0, 128, 255), -1)

                    cv.putText(frame,f"X={int(ballCenter_X)}, Y={int(ballCenter_Y)}",(text_x, text_y),cv.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)
                    cv.line(frame,(ballCenter_X,0),(ballCenter_X,frame.shape[0]),(0,128,255),2)
                    cv.line(frame,(0,ballCenter_Y),(frame.shape[1], ballCenter_Y),(0,128,255),2)
                    cv.circle(frame,(ballCenter_X,ballCenter_Y),5,(0,0,255),-1)


    mask_color=cv.cvtColor(mask_clean,cv.COLOR_GRAY2BGR)
    mergeframe=np.hstack((frame,mask_color))

    cv.imshow("Ball Tracking",mergeframe)

    # try:
    #     data=f"{ballCenter_X},{ballCenter_Y}\n"
    #     arduino.write(data.encode())   
    #     response=arduino.readline().decode().strip()
    #     print("Arduino: ",response)                          
    # except Exception as e:
    #     print("Error sending data:",e)   
    if cv.waitKey(1)&0xFF==ord('q'):
        break

videoCapture.release()
# arduino.close()
cv.destroyAllWindows()