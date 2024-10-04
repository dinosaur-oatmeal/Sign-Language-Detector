"""
ASL Hand Landmark Data Collection Tool

This program captures live video from the webcam and uses MediaPipe to detect and track a single hand.
It allows users to collect and save hand landmark data corresponding to static American Sign Language (ASL) letters.
Each saved sample includes the 3D coordinates of the hand landmarks, labeled with the specified ASL letter.
The collected data is stored in a CSV file (`StaticSignData.csv`) for use in training the machine learning model
for ASL recognition.

Dependencies:
- OpenCV (`cv2`): For video capture and display.
- MediaPipe (`mediapipe`): For hand detection and landmark extraction.
- NumPy (`numpy`): For numerical operations.
- Pandas: For data manipulation and saving data to CSV
- Other standard libraries: `os`, `warnings`.

Usage:
1. Run the program.
2. Enter the target ASL letter label (A-Z) when prompted.
3. Specify the number of samples to collect (default is 50).
4. Use the webcam feed to position your hand in view.
5. Press 's' to save the current frame's landmarks as a sample.
6. Press 'q' to quit the data collection early.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import warnings

# Suppress protobuf warnings to keep console clean
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize MediaPipe Hands for hand detection and landmark estimation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,                # Single hand for ASL letters
    min_detection_confidence=0.7,   # Minimum confidence for detection
    min_tracking_confidence=0.7     # Minimum confidence for tracking
)

# Utility for drawing hand landmarks on camera feed
mp_draw = mp.solutions.drawing_utils

def collect_landmarks(frame):
    """
    Detects hand landmarks in the given frame.

    Args:
        frame (numpy.ndarray): The current video frame in BGR format.

    Returns:
        tuple:
            - frame (numpy.ndarray): The frame with hand landmarks drawn (if detected).
            - landmark_list (list): A list of normalized (x, y, z) coordinates of the hand landmarks.
    """
    # Convert frame from BGR to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and make a list for landmark locations
    results = hands.process(img_rgb)
    landmark_list = []

    # Support for multiple hands in frame (only using 1 in practice)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the (x, y, z) coordinates of each landmark
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            
            # Draw hand landmarks and connections on camera feed
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, landmark_list

# Function to save data
def save_data(data, label):
    """
    Saves the collected landmark data with the associated label to a CSV file.

    Args:
        data (dict): A dictionary containing the label and landmark coordinates.
        label (str): The ASL letter label for the current sample.
    """
    # Convert the data dictionary into a DataFrame with one row
    new_data = pd.DataFrame([data])

    if os.path.exists('StaticSignData.csv'):
        # Read the existing CSV file if it exists
        df = pd.read_csv('StaticSignData.csv')
        
        # Append the new data to the existing DataFrame
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        # If the file doesn't exist, the new data becomes the DataFrame
        df = new_data

    # Save the updated DataFrame back to CSV without row indices
    df.to_csv('StaticSignData', index=False)

def main():
    """
    The main function to execute the data collection process.
    """
    # Initialize video capture (change output if using multiple cameras)
    cap = cv2.VideoCapture(0)
    
    # Set the camera resolution to 1920x1080 (1080p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("Starting Data Collection...")
    
    # Prompt the user to enter the target ASL letter  label (A - Z)
    while True:
        label = input("Enter the label for the current sign (A-Z): ").upper()
        if label.isalpha() and len(label) == 1:
            break
        else:
            print("Invalid label. Please enter a single alphabet character (A-Z).")
    
    # Prompt the user to enter the number of samples to collect
    while True:
        num_samples_input = input("Enter the number of samples to collect [Default: 50]: ")
        if num_samples_input == '':
            # Default number
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
    
    # Initialize sample count
    collected_samples = 0
    
    while collected_samples < num_samples:
        # Capture camera frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror the frame (more natural to use)
        frame = cv2.flip(frame, 1)

        # Detect landmarks in the frame
        frame, landmarks = collect_landmarks(frame)

        # Display current label and sample count on the frame
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Collected: {collected_samples}/{num_samples}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show annotated frame and title "Data Collection"
        cv2.imshow("Data Collection", frame)

        # Wait 1 ms for a key press (don't hold keys down)
        key = cv2.waitKey(1) & 0xFF

        # Quit program prematurely
        if key == ord('q'):
            print("Data collection terminated by user.")
            break
        elif key == ord('s'):
            # Save current landmarks to CSV if landmarks are detected
            if landmarks:
                # Prepare data dictionary with the label
                data = {'label': label}

                # Iterate through landmarks (x, y, z)
                for i in range(0, len(landmarks), 3):
                    # Assign landmarks to keys in dictionary
                    data[f'landmark_{i//3}_x'] = landmarks[i]
                    data[f'landmark_{i//3}_y'] = landmarks[i+1]
                    data[f'landmark_{i//3}_z'] = landmarks[i+2]

                # Save data to CSV file
                save_data(data, label)

                # Increment sample count
                collected_samples += 1
                print(f"Saved sample {collected_samples}/{num_samples} for label: {label}")
            else:
                # No landmarks found in saved frame
                print("No landmarks detected. Try again.")

    print(f"Data collection completed. Total samples collected for '{label}': {collected_samples}/{num_samples}")

    # Release camera feed and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run main function on program execution
    main()
