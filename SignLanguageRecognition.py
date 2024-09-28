import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

def access_camera():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,  # Set to 2 to track two hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Load the pre-trained binary model
    model = tf.keras.models.load_model('Model/sign_model_A_B.keras')
    print("Model loaded successfully.")

    # Load label encoder and scaler
    with open('Model/label_encoder_A_B.pkl', 'rb') as f:
        le = pickle.load(f)

    with open('Model/scaler_A_B.pkl', 'rb') as f:
        scaler = pickle.load(f)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks with styles
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get the label of the hand (Left/Right)
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
                prediction_prob = model.predict(landmarks_scaled)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0
                sign = le.inverse_transform([prediction])[0]

                # Display the predicted sign on the frame
                cv2.putText(frame, sign, (10, 70 + idx*30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Print the sign in the terminal
                print(f"Detected Sign ({hand_label}): {sign}")

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera()
