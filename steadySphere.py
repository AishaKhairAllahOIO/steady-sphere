import cv2 as cv
import numpy as np
import serial
import time
import threading
from playsound import playsound

sound_played=False

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
        derivative=(current_error-self.previous_error)/dt if dt>0 else 0
        output=self.Kp*current_error+self.Ki*self.integral+self.Kd*derivative
        self.previous_error=current_error
        return output

#arduino=serial.Serial('COM8',9600)
#time.sleep(2)

def getRangeHSV(BGR_color):
        BGR_color=np.uint8([[BGR_color]])
        HSV_color=cv.cvtColor(BGR_color,cv.COLOR_BGR2HSV)

        lower_color=HSV_color[0][0][0]-10,100,100
        upper_color=HSV_color[0][0][0]+10,255,255

        lower_color=np.array(lower_color,np.uint8)
        upper_color=np.array(upper_color,np.uint8)

        return lower_color,upper_color

yellow=[0,255,255]
green=[112,141,42]

lower_yellow,upper_yellow=getRangeHSV(yellow)
lower_green, upper_green =getRangeHSV(green)

print("color tracking in HSV:\n")
print("Yellow color HSV range:")
print("Lower Yellow:",lower_yellow)
print("Upper Yellow:",upper_yellow)

print("\nGreen color HSV range:")
print("Lower Green:",lower_green)
print("Upper Green:",upper_green)

videoCapture=cv.VideoCapture(0)

def ballTracker(frame,HSV_frame):
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
                ballCenter_X=int(M["m10"]/M["m00"])
                ballCenter_Y=int(M["m01"]/M["m00"])
                if radius>10:
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
               
                return mask_clean,ballCenter_X,ballCenter_Y,radius
    return mask_clean,None,None,None    

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
    return mask_clean,None,None
    
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH,640)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT,480)

pid_x=PID(Kp=0,Ki=0,Kd=0)
pid_y=PID(Kp=0,Ki=0,Kd=0)

def mapPIDtoAngle(PIDoutput):
    PIDoutput=max(min(PIDoutput,100),-100)
    angle=90+PIDoutput*0.9
    return angle

platform_center_locked=False
saved_platform_X=None
saved_platform_Y=None

while True:
    ret,frame=videoCapture.read()
    if not ret:
        break
    frame=cv.flip(frame,1)
    HSV_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    h,s,v=cv.split(HSV_frame)
    v_eq=cv.equalizeHist(v)
    HSV_frame=cv.merge([h,s,v_eq])

    if not platform_center_locked:
        mask_clean, platform_X, platform_Y=platformTracker(frame, HSV_frame)
        if platform_X is not None and platform_Y is not None:
            saved_platform_X=platform_X
            saved_platform_Y=platform_Y

    key=cv.waitKey(1)&0xFF

    if key==ord('l'):
        platform_center_locked=True
        print("Platform tracking locked")
    elif key==ord('o'):
        platform_center_locked=False
        print("Platform center unlocked.")

    print(f"Platform  center: X={saved_platform_X}, Y={saved_platform_Y}")
    print("\n")
    
    mask_clean,ballCenter_X,ballCenter_Y,radius=ballTracker(frame,HSV_frame)
    print(f"Ball  center: X={ballCenter_X}, Y={ballCenter_Y}")
    print("\n")

    if ballCenter_X is not None and ballCenter_Y is not None and not sound_played: 
        threading.Thread(target=play_sound,daemon=True).start()
        sound_played = True

    if ballCenter_X is not None and saved_platform_X is not None:
        error_x=saved_platform_X- ballCenter_X 
        print("Error X=",error_x,"\n")
        if abs(error_x)<5:
            output_x=0
            print("X Dead Zone-No correction needed.")
        else:
            output_x=pid_x.PIDcompute(error_x)
            print("pid x=",output_x)

        angle_x=mapPIDtoAngle(output_x)
        print("angle x=",angle_x)


    if ballCenter_Y is not None and saved_platform_Y is not None:
        error_y=saved_platform_Y-ballCenter_Y
        print("Error Y=",error_y)
        if abs(error_y) < 5:
            output_y = 0
            print("Y Dead Zone-No correction needed.")
        else:
            output_y = pid_y.PIDcompute(error_y)
            print("pid y=",output_y)

        angle_y=mapPIDtoAngle(output_y)
        print("angle y=",angle_y)

    mask_color=cv.cvtColor(mask_clean,cv.COLOR_GRAY2BGR)
    mergeframe=np.hstack((frame,mask_color))
    cv.imshow("Tracking",mergeframe)

    # try:
    #     data=f"{angle_x},{angle_y}\n"
    #     arduino.write(data.encode())   
    #     response=arduino.readline().decode().strip()
    #     print("position of servo x and servo y is: ",response)                          
    # except Exception as e:
    #     print("Error sending data:",e)   
   
    if key==ord('q'):
        break
    elif key==ord('w'):
        pid_x.Kp+=0.01
        print(f"Kp X={pid_x.Kp:.3f}")
    elif key==ord('e'): 
        pid_y.Kp+=0.01
        print(f"Kp Y={pid_y.Kp:.3f}")
    elif key==ord('s'): 
        pid_x.Kp=max(pid_x.Kp-0.01,0)
        print(f"Kp X={pid_x.Kp:.3f}")
    elif key==ord('d'):  
       pid_y.Kp=max(pid_y.Kp-0.01,0)
       print(f"Kp Y={pid_y.Kp:.3f}")
    elif key==ord('r'):  
       pid_x.Ki+=0.001
       print(f"Ki X={pid_x.Ki:.4f}")
    elif key==ord('t'): 
       pid_y.Ki+=0.001
       print(f"Ki Y={pid_y.Ki:.4f}")
    elif key==ord('f'): 
       pid_x.Ki=max(pid_x.Ki-0.001,0)
       print(f"Ki X={pid_x.Ki:.4f}")
    elif key==ord('g'): 
       pid_y.Ki=max(pid_y.Ki-0.001,0)
       print(f"Ki Y={pid_y.Ki:.4f}")
    elif key==ord('y'):  
       pid_x.Kd+=0.01
       print(f"Kd X={pid_x.Kd:.3f}")
    elif key==ord('u'):  
       pid_y.Kd+=0.01
       print(f"Kd Y={pid_y.Kd:.3f}")
    elif key==ord('h'):  
       pid_x.Kd=max(pid_x.Kd-0.01,0)
       print(f"Kd X={pid_x.Kd:.3f}")
    elif key==ord('j'): 
       pid_y.Kd=max(pid_y.Kd-0.01,0)
       print(f"Kd Y={pid_y.Kd:.3f}")
    elif key==ord('p'): 
        with open("pid_value.txt","w") as file:
            file.write(f"{pid_x.Kp},{pid_x.Ki},{pid_x.Kd}")
            file.write(f"{pid_x.Kp},{pid_x.Ki},{pid_x.Kd}")
        print("PID values saved to pid_value.txt")

videoCapture.release()
#arduino.close()
cv.destroyAllWindows()