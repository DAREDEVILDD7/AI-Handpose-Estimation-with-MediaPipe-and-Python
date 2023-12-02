import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup MediaPipe Hands.
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the image and draw hand landmarks.
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the processed image.
    cv2.imshow('AI Hand Pose Estimation', image)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
