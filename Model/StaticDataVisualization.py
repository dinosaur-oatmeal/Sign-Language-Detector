import pandas as pd
import numpy as np
import cv2
import math

def load_csv(csv_path):
    """
    Load the CSV file into a pandas DataFrame.
    """
    print(f"[INFO] Loading CSV file from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] CSV file loaded. Total labels: {df['label'].nunique()}")
    except FileNotFoundError:
        print(f"[ERROR] File not found at path: {csv_path}")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"[ERROR] No data found in the CSV file: {csv_path}")
        exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred while loading the CSV: {e}")
        exit(1)
    return df

def get_hand_landmark_columns():
    """
    Generate a list of hand landmark columns for MediaPipe Hands.
    Each landmark has x and y coordinates.
    """
    landmarks = []
    for i in range(21):  # MediaPipe Hands has 21 landmarks
        for coord in ['x', 'y']:
            landmarks.append(f'landmark_{i}_{coord}')
    return landmarks

def get_hand_connections():
    """
    Define the connections between landmarks as per MediaPipe Hands specification.
    This list defines which landmarks should be connected to each other.
    """
    return [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

def calculate_grid(num_items, max_columns=5):
    """
    Calculate the number of rows and columns for the grid layout based on the number of items.
    """
    columns = min(num_items, max_columns)
    rows = math.ceil(num_items / columns) if columns else 1
    return rows, columns

def compute_average_hand_landmarks(df, landmark_cols):
    """
    Compute the average x and y coordinates for each landmark per label.
    
    Args:
        df (pd.DataFrame): DataFrame containing the hand data.
        landmark_cols (list): List of landmark columns to include in the computation.
    
    Returns:
        pd.DataFrame: DataFrame with the average coordinates per label.
    """
    return df.groupby('label')[landmark_cols].mean().reset_index()

def visualize_average_hands(df, canvas_size=(320, 320), max_columns=5, labels_per_page=10):
    """
    Visualize the average hand landmarks per label in a paginated grid layout.

    Args:
        df (pd.DataFrame): DataFrame containing the hand data.
        canvas_size (tuple): Size of each individual sample's visualization window (width, height).
        max_columns (int): Maximum number of columns in the grid.
        labels_per_page (int): Number of labels to display per page.
    """
    labels = sorted(df['label'].unique())
    num_labels = len(labels)
    print(f"[INFO] Number of labels to visualize: {num_labels}")

    landmark_cols = get_hand_landmark_columns()  # Expected list of landmark columns
    available_cols = df.columns.intersection(landmark_cols)  # Get only columns present in DataFrame
    connections = get_hand_connections()

    # Calculate total number of pages
    total_pages = math.ceil(num_labels / labels_per_page)
    current_page = 0  # Zero-based indexing

    print(f"[INFO] Total pages: {total_pages}")

    # Compute average landmarks per label
    print("[INFO] Computing average landmarks per label...")
    avg_df = compute_average_hand_landmarks(df, available_cols)
    print("[INFO] Averages computed.")

    # Set up canvas dimensions
    width, height = canvas_size

    # Initialize OpenCV window
    window_name = "Average Hand Landmarks per Label (Press 'x' for Next, 'z' for Previous, 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    total_width = width * max_columns
    total_height = height * math.ceil(labels_per_page / max_columns)
    cv2.resizeWindow(window_name, total_width, total_height)

    print("[INFO] Starting visualization. Press 'x' for next page, 'z' for previous page, 'q' to quit.")

    while True:
        # Calculate the range of labels for the current page
        start_idx = current_page * labels_per_page
        end_idx = min(start_idx + labels_per_page, num_labels)
        current_page_labels = labels[start_idx:end_idx]
        current_page_count = len(current_page_labels)

        # Calculate grid layout for the current page
        rows, columns = calculate_grid(current_page_count, max_columns=max_columns)

        # Adjust total canvas size for the current page
        total_canvas_width = width * columns
        total_canvas_height = height * rows
        total_canvas = np.zeros((total_canvas_height, total_canvas_width, 3), dtype=np.uint8)

        for idx, label in enumerate(current_page_labels):
            avg_row = avg_df[avg_df['label'] == label].iloc[0]

            # Only use available landmark columns
            try:
                landmarks = avg_row[available_cols].values.reshape(-1, 2)  # Reshape to (21, 2)
            except ValueError:
                print(f"[ERROR] Reshape failed for label '{label}'. Available columns might be incomplete.")
                continue

            # Scale landmarks to fit the canvas
            landmark_points = []
            for lm in landmarks:
                x_norm, y_norm = lm
                # Clamp the normalized coordinates between 0 and 1 to avoid drawing outside the canvas
                x_norm = max(0, min(x_norm, 1))
                y_norm = max(0, min(y_norm, 1))
                x = int(x_norm * (width - 20)) + 10  # Adding padding
                y = int(y_norm * (height - 20)) + 10  # Adding padding
                landmark_points.append((x, y))

            # Create a blank canvas for the average hand
            sample_canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw connections
            for connection in connections:
                start_idx_conn, end_idx_conn = connection
                start_point = landmark_points[start_idx_conn]
                end_point = landmark_points[end_idx_conn]
                cv2.line(sample_canvas, start_point, end_point, (255, 0, 0), 2)

            # Draw landmarks
            for point in landmark_points:
                x, y = point
                cv2.circle(sample_canvas, (x, y), 5, (0, 255, 0), -1)

            # Add label text
            cv2.putText(sample_canvas, f"Label: {label}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Calculate the position on the total canvas
            row_idx = idx // columns
            col_idx = idx % columns
            x_offset = col_idx * width
            y_offset = row_idx * height

            # Place the sample canvas onto the total canvas
            total_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = sample_canvas

        # Display the total canvas
        cv2.imshow(window_name, total_canvas)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("[INFO] Visualization terminated by user.")
            break

        elif key == ord('x'):
            # Move to the next page
            if current_page < total_pages - 1:
                current_page += 1
                print(f"[INFO] Moved to page {current_page + 1}/{total_pages}.")
            else:
                print("[INFO] Reached the last page. Looping back to the first page.")
                current_page = 0

        elif key == ord('z'):
            # Move to the previous page
            if current_page > 0:
                current_page -= 1
                print(f"[INFO] Moved to page {current_page + 1}/{total_pages}.")
            else:
                print("[INFO] Already at the first page. Moving to the last page.")
                current_page = total_pages - 1

    # Close all OpenCV windows upon exit
    cv2.destroyAllWindows()

def main():
    csv_path = 'StaticSignData.csv'  # Update this path as needed
    df = load_csv(csv_path)
    visualize_average_hands(df, canvas_size=(320, 320), max_columns=5, labels_per_page=10)

if __name__ == "__main__":
    main()
