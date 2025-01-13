import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # Mediapipe for hand landmark detection
from math import hypot  # Used to calculate the Euclidean distance
import screen_brightness_control as sbc  # Library to control screen brightness
import numpy as np  # Numerical library for data manipulation

# Initialize Mediapipe Hands module
mpHands = mp.solutions.hands  # Load the hands detection solution
hands = mpHands.Hands(
    static_image_mode=False,  # False means the model works in a live video stream
    model_complexity=1,  # Complexity of the hand landmark model (higher is more accurate, slower)
    min_detection_confidence=0.75,  # Minimum confidence for detecting hands
    min_tracking_confidence=0.75,  # Minimum confidence for tracking hand landmarks
    max_num_hands=2  # Maximum number of hands to detect
)

# Utility to draw hand landmarks
Draw = mp.solutions.drawing_utils

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirrored view

    # Convert the frame to RGB as Mediapipe expects RGB input
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to detect hands and their landmarks
    Process = hands.process(frameRGB)

    # List to store the landmarks of detected hands
    landmarklist = []

    # Check if hands were detected in the frame
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:  # Iterate through each detected hand
            for _id, landmarks in enumerate(handlm.landmark):  # Get each landmark in the hand
                height, width, colour_channels = frame.shape  # Get frame dimensions
                x, y = int(landmarks.x * width), int(landmarks.y * height)  # Convert normalized coordinates to pixels
                landmarklist.append([_id, x, y])  # Store landmark ID and coordinates
            # Draw the hand landmarks and connections on the frame
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    # If landmarks are detected, perform further processing
    if landmarklist != []:
        # Coordinates of the thumb tip (landmark ID 4)
        x1, y1 = landmarklist[4][1], landmarklist[4][2]
        # Coordinates of the index finger tip (landmark ID 8)
        x2, y2 = landmarklist[8][1], landmarklist[8][2]

        # Draw circles on the thumb and index finger tips
        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)  # Green circle at thumb tip
        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)  # Green circle at index finger tip

        # Calculate the distance between thumb and index finger
        L = hypot(x2 - x1, y2 - y1)

        # Map the distance (L) to a brightness level (0 to 100%)
        blevel = np.interp(L, [15, 220], [0, 100])

        # Set the screen brightness based on the calculated level
        sbc.set_brightness(int(blevel))

    # Display the processed frame
    cv2.imshow('frame', frame)

    # Break the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xff == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
