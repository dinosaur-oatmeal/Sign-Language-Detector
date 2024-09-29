import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import warnings

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
    model_path = 'Model/signModel.keras'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        exit()

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load label encoder and scaler
    le_path = 'Model/labelEncoder.pkl'
    scaler_path = 'Model/scaler.pkl'

    if not os.path.exists(le_path) or not os.path.exists(scaler_path):
        print("Preprocessing objects not found. Please ensure 'labelEncoder.pkl' and 'scaler.pkl' exist.")
        exit()

    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Starting confidence threshold
    CONFIDENCE_THRESHOLD = 0.95

    print("Press '+' to increase the confidence threshold.")
    print("Press '-' to decrease the confidence threshold.")
    print("Press 'q' to quit.")

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
                    cv2.putText(frame, f'{hand_label} Hand', 
                                (10, 30 + idx*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (255, 0, 0), 
                                2, 
                                cv2.LINE_AA)
                else:
                    hand_label = "Unknown"

                # Extract landmarks
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                # Convert to NumPy array and reshape for the model
                landmarks = np.array(landmark_list).reshape(1, -1)
                landmarks_scaled = scaler.transform(landmarks)

                # Predict using the model
                prediction_prob = model.predict(landmarks_scaled, verbose=0)[0]
                prediction_class = np.argmax(prediction_prob)
                confidence = prediction_prob[prediction_class]
                sign = le.inverse_transform([prediction_class])[0]

                # Determine if confidence is above the threshold
                if confidence > CONFIDENCE_THRESHOLD:
                    # Assign color based on class
                    # Example: Assign unique colors for each class
                    color_dict = {'A': (255, 0, 0), 'B': (0, 255, 0), 'C': (0, 0, 255)}  # Add more as needed
                    color = color_dict.get(sign, (255, 255, 0))  # Default color if not defined

                    # Display the predicted sign on the frame
                    cv2.putText(frame, sign, (10, 70 + idx*30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2, cv2.LINE_AA)

                    # Display the confidence score
                    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 110 + idx*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    # Print the sign in the terminal
                    # print(f"Detected Sign ({hand_label}): {sign} with confidence {confidence:.2f}")
                else:
                    # Display a placeholder or confidence score
                    cv2.putText(frame, "No confident sign detected", (10, 70 + idx*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # Display the confidence score
                    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 110 + idx*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                    # Print the non-confident detection
                    # print(f"No confident sign detected (Confidence: {confidence:.2f})")
        else:
            # Display message when no hand is detected
            cv2.putText(frame, "No hand detected", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # print("No hand detected.")

        # Display the confidence threshold on the frame
        cv2.putText(frame, f'Threshold: {CONFIDENCE_THRESHOLD:.2f}', (10, 150), 
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

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera()

