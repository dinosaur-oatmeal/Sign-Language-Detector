import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import warnings

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # Single hand for ASL letters
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Function to collect landmarks
def collect_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, landmark_list

# Function to save data
def save_data(data, label):
    # Convert the single dictionary to a DataFrame with one row
    new_data = pd.DataFrame([data])

    if os.path.exists('sign_data.csv'):
        # Read the existing data
        df = pd.read_csv('sign_data.csv')
        
        # Concatenate the new data with the existing DataFrame
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        # If the file doesn't exist, the new_data becomes the DataFrame
        df = new_data

    # Save the updated DataFrame back to CSV
    df.to_csv('sign_data.csv', index=False)

def main():
    cap = cv2.VideoCapture(0)
    
    # Set the camera resolution to 1920x1080 (1080p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("Starting Data Collection...")
    
    # Prompt the user to enter the target letter
    while True:
        label = input("Enter the label for the current sign (A-Z): ").upper()
        if label.isalpha() and len(label) == 1:
            break
        else:
            print("Invalid label. Please enter a single alphabet character (A-Z).")
    
    # Prompt the user to enter the number of samples (optional)
    while True:
        num_samples_input = input("Enter the number of samples to collect [Default: 50]: ")
        if num_samples_input == '':
            num_samples = 50
            break
        elif num_samples_input.isdigit() and int(num_samples_input) > 0:
            num_samples = int(num_samples_input)
            break
        else:
            print("Invalid input. Please enter a positive integer or press Enter to use the default.")
    
    print(f"Collecting {num_samples} samples for label '{label}'.")
    print("Press 's' to save the current frame's landmarks.")
    print("Press 'q' to quit early.")
    
    collected_samples = 0
    
    while collected_samples < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame, landmarks = collect_landmarks(frame)

        # Display the count on the frame
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Collected: {collected_samples}/{num_samples}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Data collection terminated by user.")
            break
        elif key == ord('s'):
            if landmarks:
                data = {'label': label}
                for i in range(0, len(landmarks), 3):
                    data[f'landmark_{i//3}_x'] = landmarks[i]
                    data[f'landmark_{i//3}_y'] = landmarks[i+1]
                    data[f'landmark_{i//3}_z'] = landmarks[i+2]
                save_data(data, label)
                collected_samples += 1
                print(f"Saved sample {collected_samples}/{num_samples} for label: {label}")
            else:
                print("No landmarks detected. Try again.")

    print(f"Data collection completed. Total samples collected for '{label}': {collected_samples}/{num_samples}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
