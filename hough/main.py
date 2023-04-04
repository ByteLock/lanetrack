import cv2
import numpy as np

def process_frame(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for yellow and white
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 25, 255], dtype=np.uint8)
    
    # Create masks for yellow and white
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # Apply mask to the image
    masked = cv2.bitwise_and(image, image, mask=combined_mask)
    
    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    canny = cv2.Canny(blur, 50, 150)
    
    return canny


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    
    polygon = np.array([[
        (0, height),
        (width, height),
        (width//2, height//2)
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def detect_lines(img):
    return cv2.HoughLinesP(img, 1, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=100)


def calculate_lane_lines(lines):
    left_lines = []
    right_lines = []
    
    left_weights = []
    right_weights = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.average(left_lines, axis=0, weights=left_weights) if len(left_lines) > 0 else (0, 0)
    right_lane = np.average(right_lines, axis=0, weights=right_weights) if len(right_lines) > 0 else (0, 0)

    return left_lane, right_lane


def create_coordinates(image, line_parameters):
    if line_parameters[0] == 0:  # Check if the slope is zero to avoid division by zero.
        return np.array([0, 0, 0, 0])

    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    x1 = max(0, min(x1, image.shape[1] - 1))
    x2 = max(0, min(x2, image.shape[1] - 1))

    return np.array([x1, y1, x2, y2])




def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            print(f'{x1} {y1} | {x2} {y2}')
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
    return line_img



def main():
    cap = cv2.VideoCapture('./data/nissan-dat-3-24-23 - 1.mov')
    while(cap.isOpened()):
        _, frame = cap.read()
        canny_img = process_frame(frame)
        roi_img = region_of_interest(canny_img)
        detected_lines = detect_lines(roi_img)
        left_lane, right_lane = calculate_lane_lines(detected_lines)
        
        # Calculate slope
        left_slope = left_lane[0]
        right_slope = right_lane[0]
        # print("Left Slope:", left_slope, "Right Slope:", right_slope)
        
        left_line = create_coordinates(frame, left_lane)
        right_line = create_coordinates(frame, right_lane)
        lanes = [left_line, right_line]

        line_image = display_lines(frame, lanes)

        # Calculate center
        left_middle = (left_line[0] + left_line[2]) / 2
        right_middle = (right_line[0] + right_line[2]) / 2
        lane_center = (left_middle + right_middle) / 2

        # Calculate steering angle
        steering_angle = np.arctan((right_slope + left_slope) / 2)
        steering_angle_degrees = np.degrees(steering_angle)
        print("Steering Angle (degrees):", steering_angle_degrees)

        # Detect and track cars
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and track cars
        car_cascade = cv2.CascadeClassifier('./cascades/haarcascade_car.xml')

        cars = car_cascade.detectMultiScale(gray, 1.1, 9)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            car_center = (x + w // 2, y + h // 2)
            car_position = car_center[0]
            car_velocity = 0  # You need a time-based calculation to estimate the velocity
            # print("Car Position:", car_position, "Car Velocity:", car_velocity)

        # Combine and show the result
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()