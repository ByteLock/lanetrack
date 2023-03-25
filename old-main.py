import cv2
import numpy as np

# Set up video capture device
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/nissan-dat-3-24-23 - 1.mov')

# Define region of interest

# bl br tr tl
ROI = np.array([[(50, 550), (1000, 580), (850, 290), (200, 280)]], dtype=np.int32)


# Define color range for lane lines
lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

while True:
    # Read frame from video capture device
    ret, frame = cap.read()
    
    # Apply region of interest mask
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, ROI, (255, 255, 255))
    masked_frame = cv2.bitwise_and(frame, mask)
    
    # Convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)
    
    # Apply probabilistic Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    # Filter out short and horizontal lines
    if lines is not None:
        lines = [line[0] for line in lines if abs(line[0][1]-line[0][3]) < 10 and abs(line[0][0]-line[0][2]) > 50]
    
    # Compute left and right lane lines
    left_lane = None
    right_lane = None
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Compute line slope and intercept
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            # Classify line as left or right lane based on slope
            if slope < 0:
                if left_lane is None:
                    left_lane = (slope, intercept)
                else:
                    left_lane = (0.9*left_lane[0] + 0.1*slope, 0.9*left_lane[1] + 0.1*intercept)
            else:
                if right_lane is None:
                    right_lane = (slope, intercept)
                else:
                    right_lane = (0.9*right_lane[0] + 0.1*slope, 0.9*right_lane[1] + 0.1*intercept)
    
    # Draw detected lane lines on frame
    if left_lane is not None and right_lane is not None:
        y1 = 320
        y2 = 480
        left_x1 = int((y1 - left_lane[1]) / left_lane[0])
        left_x2 = int((y2 - left_lane[1]) / left_lane[0])
        right_x1 = int((y1 - right_lane[1]) / right_lane[0])
        right_x2 = int((y2 - right_lane[1]) / right_lane[0])
        cv2.line(frame, (left_x1, y1), (left_x2, y2), (0, 0, 255), 2)
        cv2.line(frame, (right_x1,y1), (right_x2, y2), (0, 0, 255), 2)
    
    # Display frame
    cv2.imshow("Lane Tracking System", frame)
    cv2.imshow("Gray Edges", edges)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and destroy windows
cap.release()
cv2.destroyAllWindows()