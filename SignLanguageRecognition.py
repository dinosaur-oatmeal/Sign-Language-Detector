import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import time
import warnings
import threading

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

def access_camera():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Set camera resolution to 1920x1080 (1080p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,  # Adjust as needed
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Load pre-trained multi-class model
    model_path = 'Model/signModel.keras'  # Updated model path
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        exit()

    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Load label encoder and scaler
    le_path = 'Model/labelEncoder.pkl'  # Updated label encoder path
    scaler_path = 'Model/scaler.pkl'    # Updated scaler path

    if not os.path.exists(le_path) or not os.path.exists(scaler_path):
        print("Preprocessing objects not found. Please ensure 'labelEncoder.pkl' and 'scaler.pkl' exist.")
        exit()

    try:
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        print("Label encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        exit()

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        exit()

    # Starting confidence threshold
    CONFIDENCE_THRESHOLD = 0.95  # Initial threshold

    # Initialize variables to store the latest landmarks
    latest_landmarks = None
    latest_landmarks_time = 0.0  # Timestamp when landmarks were last updated
    landmarks_lock = threading.Lock()  # To synchronize access to latest_landmarks

    # Initialize variables to store the last detected sign
    current_sign = ""
    current_confidence = 0.0
    sign_lock = threading.Lock()  # To synchronize access to current_sign and current_confidence

    # Event to signal the prediction thread to stop
    stop_event = threading.Event()

    # Variable to track the time when the last sign was updated
    last_sign_time = time.time()
    SIGN_DISPLAY_DURATION = 5  # seconds, optional timeout to clear sign

    print("Press '+' to increase the confidence threshold.")
    print("Press '-' to decrease the confidence threshold.")
    print("Press 'q' to quit.")

    def prediction_worker():
        """
        Worker thread that makes predictions every 1 second
        """
        nonlocal current_sign, current_confidence, last_sign_time
        while not stop_event.is_set():
            time.sleep(1)  # Wait for 1 second

            with landmarks_lock:
                if latest_landmarks is None:
                    continue  # No landmarks to predict

                # Check if landmarks are recent (e.g., within last 2 seconds)
                if time.time() - latest_landmarks_time > 2.0:
                    continue  # Landmarks are too old

                landmarks_copy = latest_landmarks.copy()

            # Predict using the model
            try:
                landmarks_scaled = scaler.transform(landmarks_copy.reshape(1, -1))
                prediction_prob = model.predict(landmarks_scaled, verbose=0)[0]
                prediction_class = np.argmax(prediction_prob)
                confidence = prediction_prob[prediction_class]
                sign = le.inverse_transform([prediction_class])[0]
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue  # Skip updating current_sign if there's an error

            # Update current_sign only if confidence is above the threshold
            if confidence > CONFIDENCE_THRESHOLD:
                with sign_lock:
                    current_sign = sign
                    current_confidence = confidence
                    last_sign_time = time.time()

                # Print the sign in the terminal
                print(f"Detected Sign: {current_sign} (Confidence: {current_confidence:.2f})")
            # Do not update current_sign if confidence is below threshold

    # Start the prediction worker thread
    worker_thread = threading.Thread(target=prediction_worker, daemon=True)
    worker_thread.start()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Flip frame horizontally for mirrored view
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image and find hands
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hands with styles
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
                        # If right hand
                        cv2.putText(frame, f'Does not track {hand_label} Hand',
                                    (10, 30 + idx*30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                        continue
                else:
                    hand_label = "Unknown"

                # Extract landmarks
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

        # Display the last detected sign on the frame
        with sign_lock:
            # Check if the sign should still be displayed (optional timeout)
            if current_sign and (time.time() - last_sign_time < SIGN_DISPLAY_DURATION):
                # Assign color based on class
                color_dict = {
                    'A': (255, 0, 0),    # Red
                    'B': (0, 255, 0),    # Green
                    'C': (0, 0, 255),    # Blue
                    'D': (255, 255, 0),  # Yellow
                    'E': (0, 255, 255),  # Cyan
                    'F': (255, 0, 255),  # Magenta
                    'G': (128, 0, 0),    # Maroon
                    'H': (0, 128, 0),    # Dark Green
                    'I': (0, 0, 128),    # Navy
                    'J': (128, 128, 0),  # Olive
                    'K': (128, 0, 128),  # Purple
                    'L': (0, 128, 128),  # Teal
                    'M': (192, 192, 192),# Silver
                    'N': (128, 128, 128),# Gray
                    'O': (255, 165, 0),  # Orange
                    'P': (255, 105, 180),# Hot Pink
                    'Q': (75, 0, 130),   # Indigo
                    'R': (240, 230, 140),# Khaki
                    'S': (173, 216, 230),# Light Blue
                    'T': (144, 238, 144),# Light Green
                    'U': (255, 182, 193),# Light Pink
                    'V': (210, 105, 30), # Chocolate
                    'W': (0, 191, 255),  # Deep Sky Blue
                    'X': (139, 69, 19),  # Saddle Brown
                    'Y': (154, 205, 50), # Yellow Green
                    'Z': (255, 69, 0)    # Red-Orange
                }
                color = color_dict.get(current_sign, (255, 255, 0))  # Default color if not defined

                # Optional: Add a semi-transparent rectangle as background for better visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 50), (400, 130), (0, 0, 0), -1)  # Black rectangle
                alpha = 0.4  # Transparency factor
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Display the predicted sign on the frame
                cv2.putText(frame, current_sign, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2, cv2.LINE_AA)

                # Display the confidence score
                cv2.putText(frame, f'Confidence: {current_confidence:.2f}', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                # Optionally, display a default message or clear the sign
                cv2.putText(frame, "No sign detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                current_sign = ""
                current_confidence = 0.0

        # Display the confidence threshold on the frame
        cv2.putText(frame, f'Threshold: {CONFIDENCE_THRESHOLD:.2f}', (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('+'):
            if CONFIDENCE_THRESHOLD < 0.95:
                CONFIDENCE_THRESHOLD += 0.05
                print(f"Confidence Threshold increased to {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord('-'):
            if CONFIDENCE_THRESHOLD > 0.05:
                CONFIDENCE_THRESHOLD -= 0.05
                print(f"Confidence Threshold decreased to {CONFIDENCE_THRESHOLD:.2f}")

    # Signal the worker thread to stop
    stop_event.set()
    # Wait for the worker thread to finish
    worker_thread.join()

    # Release resources after the loop has ended
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera()
