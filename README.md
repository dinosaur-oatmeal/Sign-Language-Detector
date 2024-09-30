# ASL Hand Sign Detection in Python

## Description

The **ASL Hand Sign Detection** project is a real-time application using Python to recognize and classify American Sign Language (ASL) hand gestures. Leveraging advanced computer vision techniques and machine learning models, this tool captures live video input, detects hand landmarks, and accurately translates gestures into corresponding ASL letters. The project integrates Text-to-Speech (TTS) functionality to relay detected signs, enhancing accessibility and user interaction audibly.

## Features

- **Custom Dataset Creation:** Collected data and curated a comprehensive dataset of American Sign Language (ASL) hand signs to train the neural network model.
- **Neural Network Training:** Designed and trained a TensorFlow-based custom neural network to accurately classify ASL letters based on hand gestures.
- **Generative AI Collaboration:** Utilized Generative AI tools to streamline the model-building process, gaining experience in designing, training, and evaluating neural networks for optimal performance.
- **Real-Time Detection:** Hosted the trained model on a local machine to detect ASL in real-time, converting hand gestures to text and audio seamlessly.
- **Multithreading Implementation:** Incorporated multithreading to handle predictions concurrently, ensuring smooth and responsive video playback.
- **Interactive User Features:** Enabled interactive functionalities allowing users to adjust confidence thresholds, accumulate detected signs, and utilize Text-to-Speech to audibly relay accumulated signs upon request.

## Algorithms and Data Structures

### Neural Network Architecture

The project employs a custom neural network built using TensorFlow, tailored for image classification tasks related to ASL hand signs.

- **Input Layer:** Accepts normalized hand landmark coordinates extracted via MediaPipe.
- **Hidden Layers:** Comprises multiple dense layers with activation functions (e.g., ReLU) to capture complex patterns in the data.
- **Output Layer:** Utilizes a softmax activation function to output probability distributions across the ASL letter classes.
- **Loss Function:** Categorical cross-entropy to measure the discrepancy between predicted and actual distributions.
- **Optimizer:** Adam optimizer for efficient gradient descent during training.

### Data Preprocessing

- **Hand Landmark Extraction:** Utilizes MediaPipe to detect and extract 21 hand landmarks from each video frame.
- **Normalization:** Scales landmark coordinates to a standardized range to improve model performance and convergence.
- **Encoding:** Applies one-hot encoding to convert categorical ASL letters into numerical representations suitable for training.

### Data Structures

- **Arrays and Matrices:** Utilized NumPy arrays for efficient storage and manipulation of hand landmark data and model weights.
- **Queues:** Implemented threading queues to manage real-time data flow between video capture, prediction, and TTS functionalities.
- **Locks:** Employed threading locks to ensure thread-safe operations when accessing and modifying shared resources like the latest landmarks and detected signs.

## Installation

Follow these steps to set up the ASL Hand Sign Detection project on your local machine:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/dinosaur-oatmeal/sign-language-detector.git
    cd Sign-Language-Detector
    ```

2. **Create a Virtual Environment (Recommended)**

    It's advisable to use a virtual environment to manage project dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    Ensure you have `pip` installed. Then, install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    *If you don't have a `requirements.txt` file, you can create one with the following content:*

    ```plaintext
    opencv-python
    mediapipe
    tensorflow
    numpy
    pyttsx3
    ```
    
4. **Download Pre-trained Model and Encoders**

    Ensure the following files are present in the `Model/` directory:

    - `signModel.keras`
    - `labelEncoder.pkl`
    - `scaler.pkl`

    *If these files are not included in the repository, run ModelTraining.py in Model/*

## Usage

1. **Run the Application**

    ```bash
    python \.SignLanguageRecognition.py
    ```

2. **Controls**

    - **'+' Key:** Increase confidence threshold.
    - **'-' Key:** Decrease confidence threshold.
    - **Spacebar:** Read aloud accumulated signs using Text-to-Speech.
    - **'q' Key:** Quit the application.

3. **Operational Flow**

    - **Start the Application:** Upon running `main.py`, the webcam feed activates, displaying real-time hand landmark detections.
    - **ASL Recognition:** Perform ASL letters with your left hand in view of the webcam. The application detects and classifies the letters in real-time.
         - The left hand was chosen over the right because it's my dominant hand.
    - **Adjust Confidence Threshold:** Use the '+' and '-' keys to adjust the sensitivity of ASL recognition based on confidence scores.
    - **Accumulate Signs:** As you perform ASL letters, they are accumulated into a string for later retrieval.
    - **Text-to-Speech:** Press the spacebar to have the accumulated signs read aloud using the TTS feature implemented via Pyttsx3.
    - **Exit:** Press the 'q' key to terminate the application gracefully, ensuring all resources are properly released.

## Technologies Used

- **Programming Language:** Python
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow, NumPy, Pickle
- **Text-to-Speech:** Pyttsx3
- **Concurrency:** Threading
- **Others:** Generative AI Tools

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) for providing robust hand landmark detection.
- [pyttsx3](https://pyttsx3.readthedocs.io/) for the Text-to-Speech functionality.
- [TensorFlow](https://www.tensorflow.org/) for the powerful machine learning framework.
- [OpenCV](https://opencv.org/) for real-time computer vision capabilities.
- Generative AI tools for assisting in the model development process.
