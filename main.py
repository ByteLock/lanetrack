import cv2
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("path/to/saved/model")

# Open the video capture device (use 0 for webcam)
cap = cv2.VideoCapture("path/to/video.mp4")

# Set the dimensions of the video frames
width = 1280
height = 720

# Loop over the frames in the video feed
while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break
        
    # Preprocess the frame and make predictions
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height)) / 255.0
    pred_lines = model.predict(np.expand_dims(img, axis=0))[0]
    
    # Draw the predicted lane markings on the frame
    for line in pred_lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    # Display the frame
    cv2.imshow('Lane Detection', frame)
    
    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the display window
cap.release()
cv2.destroyAllWindows()
