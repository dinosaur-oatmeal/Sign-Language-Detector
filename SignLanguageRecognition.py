"""
Real-Time ASL Hand Sign Detection and Display

This program captures live video from the webcam, detects hand landmarks using MediaPipe,
and classifies American Sign Language (ASL) letters in real-time using a pre-trained neural
network model. The detected sign and its confidence score are displayed on the video feed.
Additionally, users can adjust the confidence threshold for predictions and terminate the
program gracefully.

Key Features:
1. **Webcam Access and Configuration**:
    - Captures video from the default webcam.
    - Sets the camera resolution to 1920x1080 (1080p) for high-quality input.

2. **Hand Detection and Landmark Extraction**:
    - Utilizes MediaPipe Hands to detect a single hand and extract 21 hand landmarks.
    - Draws hand landmarks and connections on the video frames for visualization.

3. **Model Loading and Prediction**:
    - Loads a pre-trained multi-class classification model (`signModel.keras`).
    - Loads preprocessing objects: Label Encoder (`labelEncoder.pkl`) and Standard Scaler (`scaler.pkl`).
    - Runs predictions in a separate thread every second to ensure smooth video playback.

4. **Real-Time Display and User Interaction**:
    - Displays the detected ASL sign and its confidence score on the video frame.
    - Allows users to adjust the confidence threshold using '+' and '-' keys.
    - Provides a quit option by pressing the 'q' key.

5. **Threading and Synchronization**:
    - Implements threading to handle predictions without blocking the main video capture loop.
    - Uses threading locks to ensure thread-safe access to shared variables.

Dependencies:
- OpenCV (`cv2`): For video capture and display.
- MediaPipe (`mediapipe`): For hand detection and landmark extraction.
- NumPy (`numpy`): For numerical operations.
- TensorFlow (`tensorflow`): For loading and running the pre-trained model.
- Pickle (`pickle`): For loading preprocessing objects.
- Other standard libraries: `os`, `time`, `warnings`, `threading`.

Usage:
1. Ensure that the pre-trained model (`signModel.keras`), Label Encoder (`labelEncoder.pkl`),
   and Standard Scaler (`scaler.pkl`) are located in the `Model/` directory.
2. Run the script.
3. The webcam feed will appear with detected hand landmarks.
4. The detected ASL sign and its confidence will be displayed on the screen.
5. Press '+' to increase the confidence threshold or '-' to decrease it.
6. Press 'q' to quit the application.

Note:
- The program assumes that the hand detected is the left hand. If a right hand is detected,
  it will notify the user and skip processing.
- The confidence threshold can be adjusted to make the detection more or less stringent.
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

# Suppress protobuf warnings to keep console clean
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

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
        max_num_hands=1,  # Single hand for ASL letters
        min_detection_confidence=0.7,   # Minimum confidence for detection
        min_tracking_confidence=0.7     # Minimum confidence for tracking
    )

    # Utility for drawing hand landmarks on camera feed
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Load pre-trained multi-class classification model
    model_path = 'Model/signModel.keras'
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

    # Load the Label Encoder and Standard Scaler for preprocessing
    le_path = 'Model/labelEncoder.pkl'
    scaler_path = 'Model/scaler.pkl'

    if not os.path.exists(le_path) or not os.path.exists(scaler_path):
        print("Preprocessing objects not found. Please ensure 'labelEncoder.pkl' and 'scaler.pkl' exist.")
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

    # Initialize variables to store the latest landmarks and timestamp
    latest_landmarks = None
    latest_landmarks_time = 0.0  # Timestamp when landmarks were last updated
    landmarks_lock = threading.Lock()  # Lock to synchronize access to latest_landmarks

    # Initialize variables to store the last detected sign and its confidence
    current_sign = ""
    current_confidence = 0.0
    sign_lock = threading.Lock()  # Lock to synchronize access to current_sign and current_confidence

    # Event to signal the prediction thread to stop
    stop_event = threading.Event()

    # Variable to track the time when the last sign was updated
    last_sign_time = time.time()
    SIGN_DISPLAY_DURATION = 5  # Duration (seconds) to display detected sign

    # Print instructions to console
    print("Press '+' to increase the confidence threshold.")
    print("Press '-' to decrease the confidence threshold.")
    print("Press 'q' to quit.")

    def prediction_worker():
        """
        Worker thread that performs model predictions set amount of time based on the latest landmarks.
        Updates the current sign and its confidence if the prediction exceeds the confidence threshold.
        """
        nonlocal current_sign, current_confidence, last_sign_time
        while not stop_event.is_set():
            time.sleep(1)  # Wait for 1 second

            with landmarks_lock:
                if latest_landmarks is None:
                    continue  # No landmarks to predict

                # Check if landmarks are recent (e.g., within last 2 seconds)
                if time.time() - latest_landmarks_time > 2.0:
                    continue  # Landmarks are too old for prediction

                landmarks_copy = latest_landmarks.copy()

            # Predict using the loaded keras model
            try:
                landmarks_scaled = scaler.transform(landmarks_copy.reshape(1, -1))
                prediction_prob = model.predict(landmarks_scaled, verbose=0)[0] # Get prediction probabilities
                prediction_class = np.argmax(prediction_prob)       # Determine the class with highest probability
                confidence = prediction_prob[prediction_class]      # Confidence of the predicted class
                sign = le.inverse_transform([prediction_class])[0]  # Convert numerical label back to ASL letter
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue  # Skip updating current_sign if there's an error

            # Update current_sign only if confidence exceeds the threshold
            if confidence > CONFIDENCE_THRESHOLD:
                with sign_lock:
                    current_sign = sign
                    current_confidence = confidence
                    last_sign_time = time.time()    # Update the timestamp for display duration

                # Print the sign in the console
                print(f"Detected Sign: {current_sign} (Confidence: {current_confidence:.2f})")
            # Do not update current_sign if confidence is below threshold

    # Start the prediction worker thread as a daemon (runs in the background)
    worker_thread = threading.Thread(target=prediction_worker, daemon=True)
    worker_thread.start()

    while True:
        # Capture camera frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Mirror the frame (more natural to use)
        frame = cv2.flip(frame, 1)

        # Convert frame from BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image and find hands
        results = hands.process(img_rgb)

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

                # Update the latest_landmarks with thread safety
                with landmarks_lock:
                    latest_landmarks = np.array(landmark_list)
                    latest_landmarks_time = time.time()

        else:
            # Reset latest_landmarks when no hand is detected
            with landmarks_lock:
                latest_landmarks = None
                latest_landmarks_time = 0.0

        # Display the last detected sign and confidence in camera feed
        with sign_lock:
            # Check if the sign should still be displayed on the timeout
            if current_sign and (time.time() - last_sign_time < SIGN_DISPLAY_DURATION):

                # Define the color for the text (Red in RGB format)
                color = (255, 0, 0)

                # Semi-transparent black rectangle as a background for better text visibility
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
                # Display a default message or clear the sign when timeout occurs
                cv2.putText(frame, "No sign detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                current_sign = ""           # Reset current_sign
                current_confidence = 0.0    # Reset confidence score

        # Display the confidence threshold on the frame
        cv2.putText(frame, f'Threshold: {CONFIDENCE_THRESHOLD:.2f}', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the annotated video frame in a window titled "Camera Feed"
        cv2.imshow('Camera Feed', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit program
            print("Exiting...")
            break
        elif key == ord('+'):
            # Increase confidence up to 0.95 (default)
            if CONFIDENCE_THRESHOLD < 0.95:
                CONFIDENCE_THRESHOLD += 0.05
                print(f"Confidence Threshold increased to {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord('-'):
            # Decrease confidence
            if CONFIDENCE_THRESHOLD > 0.05:
                CONFIDENCE_THRESHOLD -= 0.05
                print(f"Confidence Threshold decreased to {CONFIDENCE_THRESHOLD:.2f}")

    # Signal the prediction worker thread to stop
    stop_event.set()
    # Wait for the worker thread to finish
    worker_thread.join()

    # Release camera feed and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run main function on program execution
    access_camera()
