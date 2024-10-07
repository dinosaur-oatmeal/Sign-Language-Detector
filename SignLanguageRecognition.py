"""
Real-Time ASL Hand Sign Detection and Display with Text-to-Speech

This program captures live video from the webcam, detects hand landmarks using MediaPipe,
and classifies American Sign Language (ASL) letters in real-time using a pre-trained neural
network model. The detected sign and its confidence score are displayed on the video feed.
Additionally, users can adjust the confidence threshold for predictions, accumulate detected
signs into a string, and have the program read the accumulated signs aloud when the spacebar
is pressed. The program can be terminated gracefully by pressing the 'q' key.

Key Features:
1. **Webcam Access and Configuration**:
    - Captures video from the default webcam.
    - Sets the camera resolution to 1920x1080 (1080p).

2. **Hand Detection and Landmark Extraction**:
    - Utilizes MediaPipe Hands to detect a single hand and extract hand landmarks.
    - Draws hand landmarks and connections on the video frames for visualization.

3. **Model Loading and Prediction**:
    - Loads a pre-trained multi-class classification model (`StaticSignModel.keras`).
    - Loads preprocessing objects: Label Encoder (`StaticLabelEncoder.pkl`) and Standard Scaler (`StaticStandardScaler.pkl`).
    - Runs predictions based on the hand being stable (not moving) over a sliding window.

4. **Real-Time Display and User Interaction**:
    - Displays the detected ASL sign and its confidence score on the video frame.
    - Allows users to adjust the confidence threshold using '+' and '-' keys.
    - Accumulates detected signs into a string.
    - Reads the accumulated signs aloud using Text-to-Speech when the spacebar is pressed.
    - Provides a quit option by pressing the 'q' key.

5. **Threading and Synchronization**:
    - Implements threading for Text-to-Speech without blocking the main video capture loop.
    - Uses threading locks to ensure thread-safe access to shared variables.

Dependencies:
- OpenCV (`cv2`): For video capture and display.
- MediaPipe (`mediapipe`): For hand detection and landmark extraction.
- NumPy (`numpy`): For numerical operations.
- TensorFlow (`tensorflow`): For loading and running the pre-trained model.
- Pickle (`pickle`): For loading preprocessing objects.
- Pyttsx3 (`pyttsx3`): For Text-to-Speech functionality.
- Other standard libraries: `os`, `time`, `warnings`, `threading`, `collections`.

Usage:
1. Ensure that the pre-trained model (`StaticSignModel.keras`), Label Encoder (`StaticLabelEncoder.pkl`),
   and Standard Scaler (`StaticStandardScaler.pkl`) are located in the `Model/` directory.
2. Run the script.
3. The webcam feed will appear with detected hand landmarks.
4. The detected ASL sign and its confidence will be displayed on the screen.
5. Press '+' to increase the confidence threshold or '-' to decrease it.
6. Press the spacebar ' ' to have the accumulated signs read aloud.
7. Press 'q' to quit the application.

Note:
- The program assumes that the hand detected is the left hand. If a right hand is detected,
  it will notify the user and skip processing.
    - This is because I am left-handed
- The confidence threshold can be adjusted to make the detection more or less stringent.
    - It is not recommended to lower them below 0.95
- Accumulated signs are read aloud and cleared when the spacebar is pressed.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import time
import warnings
import threading
from collections import deque  # Import deque for landmark buffering

import pyttsx3  # For Text-to-Speech

# Suppress protobuf warnings to keep console clean
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

class TextToSpeech:
    """
    Text-to-Speech class using pyttsx3.
    """
    def __init__(self):
        """
        Don't initialize here.
        Engine will be created in the thread.
        """
        pass

    def speak(self, text):
        """
        Convert text to speech and speak it aloud in a separate thread.

        Args:
            text (str): The text to be spoken.
        """
        # Start a new thread to run the speech (daemon runs in background)
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        """
        The thread function that creates the engine and speaks the text.

        Args:
            text (str): The text to be spoken.
        """
        # Initialize the engine in this thread
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)    # Speech rate
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()


def access_camera():
    # Initialize video capture (change output if using multiple cameras)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Set camera resolution to 1920x1080 (1080p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Initialize MediaPipe Hands for hand detection and landmark estimation
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,                # Single hand for ASL letters
        min_detection_confidence=0.7,   # Minimum confidence for detection
        min_tracking_confidence=0.7     # Minimum confidence for tracking
    )

    # Utility for drawing hand landmarks on camera feed
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Find path to pre-trained multi-class classification model (keras)
    model_path = 'Model/StaticSignModel.keras'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        exit()

    # Load keras model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    le_path = 'Model/StaticLabelEncoder.pkl'
    scaler_path = 'Model/StaticStandardScaler.pkl'

    if not os.path.exists(le_path) or not os.path.exists(scaler_path):
        print("Preprocessing objects not found. Please ensure 'StaticLabelEncoder.pkl' and 'StaticStandardScaler.pkl' exist.")
        exit()

    # Load label encoder
    try:
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        print("Label encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        exit()

    # Load standard scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        exit()

    # Initialize confidence threshold
    CONFIDENCE_THRESHOLD = 0.95 

    '''CHANGE AS NEEDED'''
    # Variables for sliding window stability detection
    WINDOW_SIZE = 30                                # Number of frames in the sliding window
    landmark_buffer = deque(maxlen=WINDOW_SIZE)     # Double-ended queue for landmarks
    AVG_MOVEMENT_THRESHOLD = 0.03                   # Hand is stable if it's below this threshold
    COOLDOWN_TIME = 0.5                             # Cooldown time before next detection can be ran (seconds)
    last_prediction_time = 0                        # Tracks last time detection was ran

    # Initialize variables to store the last detected sign and its confidence
    current_sign = ""               # Detected sign
    current_confidence = 0.0        # Confidence of that sign

    # Initialize variable to accumulate detected signs
    detected_signs = ""             # Store all detected signs
    last_appended_sign = None       # Track the last appended sign to stop doubles
    signs_lock = threading.Lock()   # Lock to synchronize access to detected_signs

    # Initialize TextToSpeech object
    tts = TextToSpeech()

    # Initialize variable to track the time when the last sign was updated
    last_sign_time = time.time()    # Tracks when the sign was posted to the screen
    SIGN_DISPLAY_DURATION = 1.75    # Duration (seconds) to display detected sign

    # Print instructions to console
    print("Press '+' to increase the confidence threshold.")
    print("Press '-' to decrease the confidence threshold.")
    print("Press ' ' (spacebar) to read the accumulated signs aloud.")
    print("Press 'q' to quit.")

    while True:
        # Capture camera frame
        ret, frame = cap.read()

        # Camera not detected
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Mirror the frame (more natural to use)
        frame = cv2.flip(frame, 1)

        # Convert frame from BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image and find hands
        results = hands.process(img_rgb)

        # No landmarks and no label
        current_landmarks = None
        hand_label = "Unknown"

        # Support for multiple hands in frame (only using 1 in practice)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks and connections on camera feed
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Hand Label (Left/Right)
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label

                    # Only process the left hand
                    if hand_label == "Left":
                        cv2.putText(frame, f'{hand_label} Hand',
                                    (10, 30 + idx*30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                    else:
                        # Notify that the right hand isn't tracked
                        cv2.putText(frame, f'Does not track {hand_label} Hand',
                                    (10, 30 + idx*30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                        continue    # Skip processing the right hand
                else:
                    hand_label = "Unknown"

                # Extract hand landmarks and flatten them into a list
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                current_landmarks = np.array(landmark_list)

                break  # Only process the first detected hand

        # Initialize flag to determine if prediction should run
        run_prediction = False

        if current_landmarks is not None:
            # Hand is found, so append current landmarks to the buffer
            landmark_buffer.append(current_landmarks)

            if len(landmark_buffer) == WINDOW_SIZE:
                # Calculate average movement over the window
                total_movement = 0.0
                for i in range(1, WINDOW_SIZE):
                    movement = np.linalg.norm(landmark_buffer[i] - landmark_buffer[i-1])
                    total_movement += movement
                avg_movement = total_movement / (WINDOW_SIZE - 1)

                # Debug: Print average movement
                # print(f"Average Movement: {avg_movement}")

                # Hand is stable
                if avg_movement < AVG_MOVEMENT_THRESHOLD:
                    current_time = time.time()
                    # Check that time is greater than the cooldown
                    if (current_time - last_prediction_time) > COOLDOWN_TIME:
                        # Run the prediction
                        run_prediction = True

                else:
                    # Reset counter if movement is detected
                    stable_frame_count = 0
        else:
            # No hand detected; clear the buffer and reset counters
            landmark_buffer.clear()
            stable_frame_count = 0
            # Delete last sign to allow double letters to be input
            last_appended_sign = None

        if run_prediction:
            # Predict using the loaded keras model
            try:
                # Scale the landmarks using the loaded scaler
                landmarks_scaled = scaler.transform(current_landmarks.reshape(1, -1))
                prediction_prob = model.predict(landmarks_scaled, verbose=0)[0]  # Get prediction probabilities
                prediction_class = np.argmax(prediction_prob)       # Determine the class with highest probability
                confidence = prediction_prob[prediction_class]      # Confidence of the predicted class
                sign = le.inverse_transform([prediction_class])[0]  # Convert numerical label back to ASL letter
            except Exception as e:
                print(f"Error during prediction: {e}")
                sign = ""
                confidence = 0.0

            # Update current_sign only if confidence exceeds the threshold
            if confidence > CONFIDENCE_THRESHOLD:
                # Check if the detected sign is different from the last appended sign
                if sign != last_appended_sign:
                    current_sign = sign
                    current_confidence = confidence
                    last_sign_time = time.time()    # Update the timestamp for display duration

                    # Append the detected sign to the accumulated string
                    with signs_lock:
                        detected_signs += sign

                    # Update the last appended sign
                    last_appended_sign = sign

                    # Print the sign in the console
                    print(f"Detected Sign: {current_sign} (Confidence: {current_confidence:.2f})")
                else:
                    print(f"Duplicate sign '{sign}' detected. Skipping append.")

                # Update last prediction time for cooldown
                last_prediction_time = time.time()

                # Clear the buffer to avoid immediate re-detection
                landmark_buffer.clear()

            # Reset stable frame count and prediction flag
            stable_frame_count = 0
            run_prediction = False

        # Display the last detected sign and confidence in camera feed
        if current_sign and (time.time() - last_sign_time < SIGN_DISPLAY_DURATION):

            # Define the color for the text (Red in BGR format)
            color = (0, 0, 255)  # OpenCV uses BGR, so red is (0, 0, 255)

            # Add a semi-transparent black rectangle as a background for better text visibility
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 50), (400, 130), (0, 0, 0), -1)  # Black rectangle
            alpha = 0.4  # Transparency factor
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Display the predicted sign on the frame
            cv2.putText(frame, current_sign, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)

            # Display the confidence score below the sign
            cv2.putText(frame, f'Confidence: {current_confidence:.2f}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            # Display default message when timeout occurs
            cv2.putText(frame, "No sign detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)     # Red color for "No sign"
            current_sign = ""                               # Reset current_sign
            current_confidence = 0.0                        # Reset confidence score

        # Display the confidence threshold on the frame
        cv2.putText(frame, f'Threshold: {CONFIDENCE_THRESHOLD:.2f}', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show annotated frame and title "Camera Feed"
        cv2.imshow('Camera Feed', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit program
            print("Exiting...")
            break
        elif key == ord('+'):
            # Increase confidence up to 0.99
            if CONFIDENCE_THRESHOLD < 0.99:
                CONFIDENCE_THRESHOLD += 0.01
                CONFIDENCE_THRESHOLD = round(CONFIDENCE_THRESHOLD, 2)  # Round to 2 decimal places
                print(f"Confidence Threshold increased to {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord('-'):
            # Decrease confidence down to 0.05
            if CONFIDENCE_THRESHOLD > 0.05:
                CONFIDENCE_THRESHOLD -= 0.01
                CONFIDENCE_THRESHOLD = round(CONFIDENCE_THRESHOLD, 2)  # Round to 2 decimal places
                print(f"Confidence Threshold decreased to {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord(' '):  # Spacebar pressed
            # Read the accumulated signs aloud
            with signs_lock:
                if detected_signs.strip():  # Check if there's anything to speak
                    print(f"Reading aloud: {detected_signs.strip()}")
                    tts.speak(detected_signs.strip())
                    detected_signs = ""  # Reset the accumulated signs after speaking
                else:
                    print("No signs to read.")

    '''
    Graceful Shutdown
    '''
    # Release camera feed and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run main function on program execution
    access_camera()