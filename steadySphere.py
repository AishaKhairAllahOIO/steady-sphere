import cv2 as cv
import numpy as np
# import serial
import time
import threading
from playsound import playsound

class LogColor:
    RESET   = "\033[0m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    PURPLE  = "\033[95m"
    GRAY    = "\033[90m"
    PINK    = "\033[38;5;213m" 
    PINK_M   ="\033[95m" 

def log_msg(message,level="INFO"):
    colors={
        "SUCCESS": LogColor.PINK_M,     
        "INFO": LogColor.PINK,     
        "ERROR": LogColor.PURPLE,     
        "CRITICAL": LogColor.BLUE,  
        "PLATFORM":LogColor.GREEN, 
        "BALL": LogColor.YELLOW,    
      
    }
    prefixes = {
        "SUCCESS": "[SUCCESS]",
        "INFO": "[INFO]",
        "ERROR": "[ERROR]",
        "CRITICAL": "[MAZE]",
        "PLATFORM": "[PLATFORM]",
        "BALL": "[BALL]",
    }
    level_upper=level.upper()
    color=colors.get(level_upper,"\033[0m")
    pre=prefixes.get(level_upper,f"[{level_upper}]")
    reset="\033[0m"
    print(f"{color}{pre}: {message}{reset}")



class SoundManager:
    def __init__(self,sound_path):
        self.sound_played=False
        self.sound_path=sound_path

    def play_sound(self):
        try:
            playsound(self.sound_path)
        except Exception as e:
            log_msg(f"Sound play error: {e}", "ERROR")
    def try_play(self):
        if not self.sound_played:
            threading.Thread(target=self.play_sound,daemon=True).start()
            self.sound_played=True


class ColorTracker:
    def __init__(self):
        pass

    def getRangeHSV(self,BGR_color):
        BGR_color=np.uint8([[BGR_color]])
        HSV_color=cv.cvtColor(BGR_color,cv.COLOR_BGR2HSV)
        lower_color=HSV_color[0][0][0]-20,100,100
        upper_color=HSV_color[0][0][0]+20,255,255
        lower_color=np.array(lower_color,np.uint8)
        upper_color=np.array(upper_color,np.uint8)

        return lower_color,upper_color
    
    def prepareFrameForTrackingByHSV(self,frame):
        frame=cv.flip(frame,1)
        HSV_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        h,s,v=cv.split(HSV_frame)
        v_eq=cv.equalizeHist(v)
        HSV_frame=cv.merge([h,s,v_eq])

        return frame,HSV_frame

    def prepareFrameForContour(self,HSV_frame,lower_color,upper_color,MORPH_TYPE):
        mask=cv.inRange(HSV_frame,lower_color,upper_color)
        blurred=cv.bilateralFilter(mask,9,75,75)
        kernel=cv.getStructuringElement(MORPH_TYPE,(7,7))
        mask_clean=cv.morphologyEx(blurred,cv.MORPH_OPEN,kernel)
        mask_clean=cv.morphologyEx(mask_clean,cv.MORPH_CLOSE,kernel)
        contours, _=cv.findContours(mask_clean,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        return mask_clean,contours
   
    def track_ball_position(self,frame,HSV_frame,lower_color,upper_color):
        ballCenter_X,ballCenter_Y=None,None
        mask_clean,contours=self.prepareFrameForContour(HSV_frame,lower_color,upper_color,cv.MORPH_ELLIPSE)

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
                    if radius is not None and radius>10:
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
               
                    return mask_clean,ballCenter_X,ballCenter_Y              
        return mask_clean,None,None 

        
    def track_rectangle_position(self,frame,HSV_frame,lower_color,upper_color):
        platform_X,platform_Y=None,None
        mask_clean,contours=self.prepareFrameForContour(HSV_frame,lower_color,upper_color,cv.MORPH_RECT)
        
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
                
                    return mask_clean,platform_X, platform_Y
        return mask_clean,None,None
    

    def order_points(self,points):
        rect=np.zeros((4,2),dtype="float32")
        s=points.sum(axis=1)
        diff=np.diff(points,axis=1)
        rect[0]=points[np.argmin(s)]
        rect[2]=points[np.argmax(s)]
        rect[1]=points[np.argmin(diff)]
        rect[3]=points[np.argmax(diff)]

        return rect
    
    def find_maze_blue(self,frame,HSV_frame,lower_color, upper_color):
        mask_clean,contours=self.prepareFrameForContour(HSV_frame,lower_color,upper_color,cv.MORPH_RECT)

        contours, _=cv.findContours(mask_clean,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None,None,None

        best_contour=max(contours,key=cv.contourArea)
        area=cv.contourArea(best_contour)
        if area<1000:
            return None,None,None

        epsilon=0.02*cv.arcLength(best_contour,True)
        approx=cv.approxPolyDP(best_contour,epsilon,True)

        if len(approx)!=4:
            return None,None,None

        points=approx.reshape(4, 2)
        rect=self.order_points(points)

        width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
        height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M=cv.getPerspectiveTransform(rect,dst)
        warped=cv.warpPerspective(frame,M,(width,height))

        return warped,rect,mask_clean
    
    def maze_to_matrix(self,maze_img,grid_size):
        hsv = cv.cvtColor(maze_img, cv.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv.inRange(hsv, lower_black, upper_black)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv.inRange(hsv, lower_green, upper_green)

        h, w = maze_img.shape[:2]
        rows, cols = grid_size
        cell_h, cell_w = h // rows, w // cols

        maze_matrix = np.zeros((rows, cols), dtype=np.int32)

        for i in range(rows):
            for j in range(cols):
                y1, y2 = i*cell_h, (i+1)*cell_h
                x1, x2 = j*cell_w, (j+1)*cell_w

                cell_black = np.sum(mask_black[y1:y2, x1:x2]) / 255
                cell_green = np.sum(mask_green[y1:y2, x1:x2]) / 255

                maze_matrix[i, j] = 1 if cell_black > cell_green else 0

        return maze_matrix
    
    @staticmethod
    def solve_maze(maze_matrix, start=(0,0), end=None):
        from queue import Queue
        rows, cols = maze_matrix.shape
        if end is None:
            end = (rows-1, cols-1)

        maze_copy = [[' ' if maze_matrix[r,c]==0 else '#' for c in range(cols)] for r in range(rows)]

        graph = {}
        for r in range(rows):
            for c in range(cols):
                if maze_copy[r][c] != '#':
                    adj = []
                    if r+1<rows and maze_copy[r+1][c] != '#': adj.append((r+1,c))
                    if r-1>=0 and maze_copy[r-1][c] != '#': adj.append((r-1,c))
                    if c+1<cols and maze_copy[r][c+1] != '#': adj.append((r,c+1))
                    if c-1>=0 and maze_copy[r][c-1] != '#': adj.append((r,c-1))
                    graph[(r,c)] = adj

        visited = set()
        queue = Queue()
        queue.put([start])
        path_found = []

        while not queue.empty():
            path = queue.get()
            node = path[-1]
            if node == end:
                path_found = path
                for r, c in path:
                    if maze_copy[r][c] == ' ':
                        maze_copy[r][c] = 'p'
                break
            if node not in visited:
                visited.add(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        queue.put(path + [neighbor])

        return maze_copy, path_found

    

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


class GimbalController:
    def __init__(self,pid):
        self.pid=pid

    def mapPIDtoAngle(self,PIDoutput,pid_min,pid_max,angle_min,angle_max):
        PIDoutput=max(min(PIDoutput,pid_max),pid_min)
        m=(angle_max-angle_min)/(pid_max-pid_min)
        b=angle_min-m*pid_min
        angle=m*PIDoutput+b
        return angle
    
# arduino=serial.Serial('COM12',9600)
# time.sleep(2)    

yellow=[0,255,255]
green=[192,142,255]
blue=[177,118,18]


colorTracker=ColorTracker()
lower_yellow,upper_yellow=colorTracker.getRangeHSV(yellow)
lower_green,upper_green=colorTracker.getRangeHSV(green)
lower_blue,upper_blue=colorTracker.getRangeHSV(blue)

log_msg("Color use for track in HSV:","INFO")
log_msg("Yellow color HSV range:","BALL")
log_msg(f"Lower Yellow: {lower_yellow}","BALL")
log_msg(f"Upper Yellow: {upper_yellow}","BALL")
log_msg("Green color HSV range:","PLATFORM")
log_msg(f"Lower Green: {lower_green}","PLATFORM")
log_msg(f"Upper Green: {upper_green}","PLATFORM")
log_msg("Blue color HSV range:","MAZE")
log_msg(f"Lower Blue: {lower_blue}","MAZE")
log_msg(f"Upper Blue: {upper_blue}","MAZE")
print("\n")

videoCapture=cv.VideoCapture(1,cv.CAP_DSHOW)
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH,640)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT,480)

pid_x=PID(Kp=0.2,Ki=0.004,Kd=0.07)
pid_y=PID(Kp=0.09999999999999996,Ki=0.007,Kd=0.03)

gimbalController_x=GimbalController(pid_x)
gimbalController_y=GimbalController(pid_y)

sound=SoundManager("C:/Users/ASUS/steady-sphere/sound/sound.m4a")

platform_center_locked=False
saved_platform_X=None
saved_platform_Y=None
ballCenter_X=None
ballCenter_Y=None


while True:
    ret,frame=videoCapture.read()
    if not ret:
        break

    frame,HSV_frame=colorTracker.prepareFrameForTrackingByHSV(frame)
    maze_img,maze_rect,maze_mask=colorTracker.find_maze_blue(frame,HSV_frame,lower_blue,upper_blue)


    if not platform_center_locked:
        mask_clean,platform_X,platform_Y=colorTracker.track_rectangle_position(frame,HSV_frame,lower_green,upper_green)
        if platform_X is not None and platform_Y is not None:
            saved_platform_X=platform_X
            saved_platform_Y=platform_Y

    key=cv.waitKey(1)&0xFF

    if key==ord('l'):
        platform_center_locked=True
        log_msg("Platform tracking locked","INFO")
    elif key==ord('o'):
        platform_center_locked=False
        log_msg("Platform tracking unlocked","INFO")
    log_msg(f"Platform  center: X={saved_platform_X}, Y={saved_platform_Y}", "PLATFORM")
    
    mask_clean,ballCenter_X,ballCenter_Y=colorTracker.track_ball_position(frame,HSV_frame,lower_yellow,upper_yellow)
    log_msg(f"Ball  center: X={ballCenter_X}, Y={ballCenter_Y}", "BALL")
    print("\n")

    if ballCenter_X is not None and ballCenter_Y is not None: 
        sound.try_play()

    if ballCenter_X is not None and saved_platform_X is not None:
        error_x=saved_platform_X-ballCenter_X 
        log_msg(f"Error X={error_x}", "ERROR")
        print("\n")
        if abs(error_x)<5:
            output_x=0
            log_msg("X Dead Zone-No correction needed.","INFO")
        else:
            output_x=pid_x.PIDcompute(error_x)
            log_msg(f"pid x={output_x}", "INFO")

        angle_x=gimbalController_x.mapPIDtoAngle(output_x,-100,100,125,155)
        log_msg(f"angle x={angle_x}", "INFO")

    if ballCenter_Y is not None and saved_platform_Y is not None:
        error_y=saved_platform_Y-ballCenter_Y
        log_msg(f"Error Y={error_y}", "ERROR")
        if abs(error_y) <5:
            output_y = 0
            log_msg("Y Dead Zone-No correction needed.","INFO")
        else:
            output_y = pid_y.PIDcompute(error_y)
            log_msg(f"pid y={output_y}", "INFO")

        angle_y=gimbalController_y.mapPIDtoAngle(output_y,-100,100,140,180)
        log_msg(f"angle y={angle_y}", "INFO")


    # if 'angle_x' in locals() and 'angle_y' in locals():
    #     try:
    #         data = f"{angle_x},{angle_y}\n"
    #         arduino.write(data.encode())
    #         response=arduino.readline().decode(errors='ignore').strip()
    #         if response:
    #             log_msg(f"Arduino response: {response}","INFO")
    #     except Exception as e:
    #             log_msg(f"Failed to send data to Arduino: {e}","ERROR")

    mask_color=cv.cvtColor(mask_clean,cv.COLOR_GRAY2BGR)
    if maze_mask is not None and maze_mask.size > 0:
        mask_blue=cv.cvtColor(maze_mask,cv.COLOR_GRAY2BGR)
        mask_color=cv.bitwise_or(mask_color,mask_blue)

    else:
        mask_blue = np.zeros_like(frame)

    mergeframe=np.hstack((frame,mask_color))
    cv.imshow("Tracking",mergeframe)
       
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
    elif key == ord('z'):
       pid_x.Kp = 0
       pid_x.Ki = 0
       pid_x.Kd = 0
       pid_y.Kp = 0
       pid_y.Ki = 0
       pid_y.Kd = 0
       print("PID values reset to 0")
    elif key==ord('p'): 
        with open("pid_value.txt","w") as file:
            file.write(f"{pid_x.Kp},{pid_x.Ki},{pid_x.Kd}\n")
            file.write(f"{pid_y.Kp},{pid_y.Ki},{pid_y.Kd}\n")
        print("PID values saved to pid_value.txt")
    elif key==ord('m'):
        maze_img,maze_rect,_=colorTracker.find_maze_blue(frame,HSV_frame,lower_blue,upper_blue)
        if maze_img is not None:
            cv.imshow("Cropped Maze", maze_img)
            log_msg("Maze cropped!", "CRITICAL")
            save_path = "cropped_maze.png" 
            cv.imwrite(save_path, maze_img)
            log_msg(f"Maze saved to {save_path}", "INFO")
            maze_matrix = colorTracker.maze_to_matrix(maze_img, grid_size=(50,50))
            maze_solved, path = colorTracker.solve_maze(maze_matrix)
           
            maze_sol="maze_solution.txt"
            with open(maze_sol, 'w') as f:
                for row in maze_solved:
                    f.write(''.join(row) + '\n')


            save_path_txt = "maze_matrix.txt"
            with open(save_path_txt, "w") as f:
                for row in maze_matrix:
                    row_str = ",".join(str(cell) for cell in row)
                    f.write(row_str + "\n")

        else:
            log_msg("No valid maze found to crop!", "ERROR")
    

videoCapture.release()
# arduino.close()
cv.destroyAllWindows()