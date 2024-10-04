"""
ASL Hand Landmark Data Collection Tool

This program captures live video from the webcam and uses MediaPipe to detect and track a single hand.
It allows users to collect and save hand landmark data corresponding to dynamic American Sign Language (ASL) letters,
    specifically, 'J' and 'Z'.
Each saved sample includes 40 frames of the hand's 3D coordinate landmarks, labeled with the specified ASL letter.
The collected data is stored in a CSV file (`DynamicSignData.csv`) for use in training the machine learning model
for ASL recognition.

Dependencies:
- OpenCV (`cv2`): For video capture and display.
- MediaPipe (`mediapipe`): For hand detection and landmark extraction.
- NumPy (`numpy`): For numerical operations.
- Pandas: For data manipulation and saving data to CSV
- Other standard libraries: `os`, `warnings`.

Usage:
1. Run the program.
2. Enter the target ASL letter label (J or Z) when prompted.
3. Specify the number of sequences to collect (default is 20 to save on storage).
4. Use the webcam feed to position your hand in view.
5. Press 's' to save the current sequence's landmarks as a sample.
6. Press 'q' to quit the data collection early.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import warnings
import time

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
def save_data(data, csv_path='DynamicSignData.csv'):
    """
    Saves the collected landmark data with the associated label and frame number to a CSV file.

    Args:
        data (list): A list of dictionaries containing the label, frame number, and landmark coordinates for each frame in the sequence.
        csv_path (str): Path to the CSV file where data will be saved.
    """
    # Convert the list of dictionaries into a DataFrame
    new_data = pd.DataFrame(data)

    # If the CSV exists, append without headers; else, write with headers
    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"Sequence saved to {csv_path}")

def main():
    """
    The main function to execute the data collection process.
    """
    # Initialize video capture (change output if using multiple cameras)
    cap = cv2.VideoCapture(0)

    # Set the camera resolution to 1280x720 for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting Data Collection...")

    # Prompt the user to enter the target ASL letter label (J or Z)
    while True:
        label = input("Enter the label for the current sign (J or Z): ").upper()
        if label.isalpha() and len(label) == 1:
            break
        else:
            print("Invalid label. Please enter a single alphabet character (J or Z).")

    # Prompt the user to enter the number of sequences to collect
    while True:
        num_sequences_input = input("Enter the number of sequences to collect [Default: 20]: ")
        if num_sequences_input == '':
            # Default number (save on storage space because every sample is 40 frames)
            num_sequences = 20
            break
        elif num_sequences_input.isdigit() and int(num_sequences_input) > 0:
            num_sequences = int(num_sequences_input)
            break
        else:
            print("Invalid input. Please enter a positive integer or press Enter to use the default.")

    # Set the number of frames for dynamic motion to capture for each sample
    sequence_length = 40

    print(f"\nCollecting {num_sequences} sequences for label '{label}'.")
    print("Ensure your hand is clearly visible in the webcam feed.")
    print("Press 's' to start capturing a sequence.")
    print("Press 'q' to quit early.\n")

    # Initialize sequence count
    collected_sequences = 0

    while collected_sequences < num_sequences:
        # Capture camera frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror the frame (more natural to use)
        frame = cv2.flip(frame, 1)

        # Detect landmarks in the frame
        frame, landmarks = collect_landmarks(frame.copy())

        # Display current label and sequence count on the frame
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Collected: {collected_sequences}/{num_sequences}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 's' to start capturing", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Show annotated frame and title "Dynamic Data Collection"
        cv2.imshow("Dynamic Data Collection", frame)

        # Wait 1 ms for a key press (don't hold keys down)
        key = cv2.waitKey(1) & 0xFF

        # Quit program prematurely
        if key == ord('q'):
            print("Data collection terminated by user.")
            break
        # Record motion to be saved to CSV file
        elif key == ord('s'):
            # Wait until landmarks are detected before starting the sequence
            print("Initializing sequence capture. Please hold your hand steady...")
            start_time = time.time()
            while True:
                # Read the frame from the camera
                ret, frame = cap.read()

                # Check to see if frame successfully captured
                if not ret:
                    print("Failed to grab frame during initialization.")
                    break

                # Mirror the frame (more natural to use)
                frame = cv2.flip(frame, 1)

                # Detect landmarks in the frame
                frame, landmarks = collect_landmarks(frame.copy())

                if landmarks:
                    cv2.imshow("Dynamic Data Collection", frame)

                    # Wait to render the frame
                    cv2.waitKey(1)
                    break

                # Hand not detected in the frame
                else:
                    cv2.putText(frame, "No hand detected. Please position your hand.", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Dynamic Data Collection", frame)

                    # Shortcut to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Data collection terminated by user during initialization.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            # Empty list to store data from each frame in sequence
            sequence_data = []
            print(f"Starting sequence {collected_sequences + 1}/{num_sequences}...")

            # Loop until all frames are filled
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                # Check to see if frame successfully captured
                if not ret:
                    print("Failed to grab frame during sequence capture.")
                    break

                # Mirror the frame (more natural to use)
                frame = cv2.flip(frame, 1)

                # Detect landmarks in the frame
                frame, landmarks = collect_landmarks(frame.copy())

                # Display progress on the frame
                cv2.putText(frame, f"Capturing frame {frame_num + 1}/{sequence_length}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

                # Show the frame while capturing the sequence
                cv2.imshow("Dynamic Data Collection", frame)

                # Shortcut to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Data collection terminated by user during sequence capture.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Save landmarks if detected
                if landmarks:
                    # Initialize a dictionary with the current label and frame number
                    data = {'label': label, 'frame_num': frame_num}

                    # Add data to ditctionary for each frame [x, y, z]
                    for i in range(0, len(landmarks), 3):
                        data[f'landmark_{i//3}_x'] = landmarks[i]
                        data[f'landmark_{i//3}_y'] = landmarks[i+1]
                        data[f'landmark_{i//3}_z'] = landmarks[i+2]
                    sequence_data.append(data)

                # Landmarks not detected
                else:
                    # Handle missing landmarks (e.g., append NaNs or skip)
                    print(f"No landmarks detected in frame {frame_num + 1}.")
                    data = {'label': label, 'frame_num': frame_num}

                    # Assign NaN to [x, y, z]
                    for i in range(0, 21 * 3, 3):
                        data[f'landmark_{i//3}_x'] = np.nan
                        data[f'landmark_{i//3}_y'] = np.nan
                        data[f'landmark_{i//3}_z'] = np.nan
                    sequence_data.append(data)

            # Save the sequence to CSV file
            save_data(sequence_data, csv_path='DynamicSignData.csv')
            collected_sequences += 1
            print(f"Saved sequence {collected_sequences}/{num_sequences} for label: {label}\n")

    print(f"\nData collection completed. Total sequences collected for '{label}': {collected_sequences}/{num_sequences}")

    # Release camera feed and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run main function on program execution
    main()