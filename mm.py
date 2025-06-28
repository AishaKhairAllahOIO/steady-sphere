import cv2 as cv
import numpy as np

def find_maze_blue(frame, hsv_frame, lower_blue, upper_blue):

    mask = cv.inRange(hsv_frame, lower_blue, upper_blue)
    blurred = cv.bilateralFilter(mask, 9, 75, 75)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    clean_mask = cv.morphologyEx(blurred, cv.MORPH_OPEN, kernel)
    clean_mask = cv.morphologyEx(clean_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    best_contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(best_contour)
    if area < 5000:
        return None, None, None

    epsilon = 0.02 * cv.arcLength(best_contour, True)
    approx = cv.approxPolyDP(best_contour, epsilon, True)

    if len(approx) != 4:
        return None, None, None

    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(frame, M, (width, height))
 
 


    return warped, rect, clean_mask

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
cap = cv.VideoCapture(0)

lower_blue = np.array([100, 100, 50])
upper_blue = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame,1)  
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    maze_img, corners, mask = find_maze_blue(frame, hsv, lower_blue, upper_blue)

    if maze_img is not None:
        cv.imshow("Cropped Maze", maze_img)

    cv.imshow("Original", frame)
    cv.imshow("Blue Mask", mask if mask is not None else np.zeros_like(frame[:, :, 0]))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
