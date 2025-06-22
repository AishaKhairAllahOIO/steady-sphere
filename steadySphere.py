import cv2 as cv
import numpy as np
import serial
import time
from playsound import playsound
import threading


sound_played = False
def play_sound():
    playsound("C:/Users/ASUS/steady-sphere/sound/sound.mp3")
class PID:
    def __init__(self,Kp,Ki,Kd):
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
        self.integral=0
        self.previous_error=0
        self.last_time=time.time()

    def PIDcompute(self,current_error):
        current_time=time.time()
        dt=current_time-self.last_time if self.last_time else 0
        self.last_time=current_time

        self.integral +=current_error*dt
        derivative=(current_error-self.previous_error)/ dt if dt>0 else 0
        output=self.Kp*current_error+self.Ki*self.integral+self.Kd*derivative
        self.previous_error=current_error
        return output

# arduino=serial.Serial('COM8',9600)
# time.sleep(2)

def getRangeHSV(BGR_color):
        BGR_color=np.uint8([[BGR_color]])
        HSV_color=cv.cvtColor(BGR_color,cv.COLOR_BGR2HSV)

        lower_color=HSV_color[0][0][0]-10,100,100
        upper_color=HSV_color[0][0][0]+10,255,255

        lower_color=np.array(lower_color,np.uint8)
        upper_color=np.array(upper_color,np.uint8)

        return lower_color,upper_color


yellow=[0,255,255]
green=[0,255,0]#112,141,42

lower_yellow,upper_yellow=getRangeHSV(yellow)
lower_green, upper_green =getRangeHSV(green)

print("Yellow color HSV range:")
print("Lower bound:", lower_yellow)
print("Upper bound:", upper_yellow)

print("\nGreen color HSV range:")
print("Lower bound:", lower_green)
print("Upper bound:", upper_green)

videoCapture=cv.VideoCapture(0)

def ballTracker(frame, HSV_frame):
    ballCenter_X,ballCenter_Y=None,None
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
            if circularity>0.5 and area>300: 
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
               
                return mask_clean, ballCenter_X, ballCenter_Y,radius
    return mask_clean, None, None,None    

def platformTracker(frame,HSV_frame):
    platform_X,platform_Y=None,None
    mask=cv.inRange(HSV_frame,lower_green,upper_green)
    blurred=cv.bilateralFilter(mask,9,75,75)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(7,7))
    mask_clean=cv.morphologyEx(blurred,cv.MORPH_OPEN,kernel)
    mask_clean=cv.morphologyEx(mask_clean,cv.MORPH_CLOSE,kernel)
    contours,_=cv.findContours(mask_clean,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if contours:
        best_contour=max(contours,key=cv.contourArea)
        area=cv.contourArea(best_contour)
        if area>1000:
            rect=cv.minAreaRect(best_contour)
            box=cv.boxPoints(rect)
            box=np.int32(box)

            M=cv.moments(best_contour)
            if M["m00"]>0:
                platform_X=int(M["m10"]/M["m00"])
                platform_Y=int(M["m01"]/M["m00"])

                cv.drawContours(frame,[box],0,(0,0,0),4)
                cv.line(frame,(platform_X,0),(platform_X,frame.shape[0]),(128,128,128),2)
                cv.line(frame,(0,platform_Y),(frame.shape[1],platform_Y),(128,128,128),2)
                cv.circle(frame,(platform_X,platform_Y),5,(0,0,0),-1)

                text_size,_=cv.getTextSize( f"X={platform_X}, Y={platform_Y}",cv.FONT_HERSHEY_SIMPLEX,0.6,2)
                text_x=platform_X+10
                text_y=platform_Y+10

                cv.rectangle(frame,(text_x-5,text_y-text_size[1]-5),(text_x+text_size[0]+5,text_y+5),(192,192,192),-1)
                cv.putText(frame, f"X={platform_X}, Y={platform_Y}",(text_x, text_y),cv.FONT_HERSHEY_SIMPLEX,0.6,(64,64,64),2)
               
                return mask_clean, platform_X, platform_Y
    return mask_clean, None, None
    
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH,640)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT,480)

pid_x=PID(Kp=0,Ki=0,Kd=0)
pid_y=PID(Kp=0,Ki=0,Kd=0)


def mapPIDtoPWM(output):
    output=max(min(output,100),-100)
    pwm=1500+output*3
    pwm=int(max(min(pwm,1800), 1200))
    return pwm


while True:
    ret,frame=videoCapture.read()
    if not ret:
        break
    frame=cv.flip(frame,1)
    HSV_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    h,s,v=cv.split(HSV_frame)
    v_eq=cv.equalizeHist(v)
    HSV_frame=cv.merge([h,s,v_eq])

    mask_clean,ballCenter_X,ballCenter_Y,radius=ballTracker(frame,HSV_frame)
    print("ballCenter X=",ballCenter_X)    
    print("ballCenter Y=",ballCenter_Y)    
    print("Radius=",radius,"\n")
    mask_clean_platform,platform_X,platform_Y=platformTracker(frame,HSV_frame)
    print("platform X=",platform_X)    
    print("platform Y=",platform_Y,"\n")

    if ballCenter_X is not None and ballCenter_Y is not None and not sound_played: 
        threading.Thread(target=play_sound,daemon=True).start()
        sound_played = True



    if ballCenter_X is not None and platform_X is not None:
        error_x=platform_X- ballCenter_X 
        print("Error X=",error_x)
        if abs(error_x)<5:
            output_x=0
            print("X Dead Zone - No correction needed.")
        else:
            output_x = pid_x.PIDcompute(error_x)
            pwm_x=mapPIDtoPWM(output_x)


    if ballCenter_Y is not None and platform_Y is not None:
        error_y=platform_Y-ballCenter_Y
        print("Error Y=",error_y)
        if abs(error_y) < 5:
            output_y = 0
            print("Y Dead Zone - No correction needed.")
        else:
            output_y = pid_y.PIDcompute(error_y)
            pwm_y=mapPIDtoPWM(output_y)



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