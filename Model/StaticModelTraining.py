"""
ASL Hand Sign Classification Model Trainer (Static Signs)

This program trains a neural network model to recognize American Sign Language (ASL)
    letters based on hand landmark data.
It performs the following steps:

1. **Data Loading and Inspection**:
    - Loads the collected hand landmark data from a CSV file (`StaticSignData.csv`).
    - Displays the first few rows, class distribution, and checks for missing values.

2. **Data Preprocessing**:
    - Encodes the categorical labels (ASL letters) into numerical format.
    - Applies one-hot encoding for multi-class classification.
        - This is needed for more than just 'A' and 'B' to be classified.
    - Scales the feature data to standardize the input for the neural network.

3. **Data Splitting**:
    - Splits the dataset into training and testing sets to evaluate the model's performance.
    - 80% training and 20% testing

4. **Model Building**:
    - Constructs a sequential neural network with multiple dense layers and dropout for regularization.
    - Compiles the model with appropriate loss function and optimizer.

5. **Model Training**:
    - Trains the model using the training data.
    - Utilizes callbacks like EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model.
        - Allows model training to "end early"

6. **Model Evaluation**:
    - Evaluates the trained model on the test set.
    - Generates a classification report and confusion matrix to assess performance.
        - Helps determine if the dataset needs to be altered.

7. **Saving the Model and Preprocessing Objects**:
    - Saves the trained model, label encoder, and scaler for future use in predictions.

Dependencies:
- pandas, numpy: Data manipulation and numerical operations.
- scikit-learn: Data preprocessing and model evaluation.
- TensorFlow/Keras: Building and training the neural network.
- matplotlib, seaborn: Data visualization.
- pickle: Saving preprocessing objects.

Usage:
1. Ensure that `StaticSignData.csv` is present in the same directory.
2. Run the script.
3. The script will output various inspection details, train the model, display training history plots,
    evaluate the model, and save the necessary objects.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_and_inspect_data(csv_path='StaticSignData.csv'):
    """
    Load the dataset and perform initial inspections.

    Args:
        csv_path (str): Path to the CSV file containing the sign data.

    Returns:
        pandas.DataFrame: The loaded and cleaned DataFrame.
    """
    # Load the dataset from the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Display the first five rows to show data structure
    print("First five rows of the dataset:")
    print(df.head())
    
    # Check distribution of each class (ASL letter)
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Check for missing values in each column
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Handle missing values if applicable (remove those rows)
    if df.isnull().values.any():
        print("\nHandling missing values by dropping rows with missing data.")
        df = df.dropna()
    
    return df

def preprocess_data(df):
    """
    Encodes the categorical labels into a numerical format and scales
    the feature data to standardize the input for the neural network.

    Args:
        df (pandas.DataFrame): The DataFrame containing features and labels.

    Returns:
        tuple:
            - X_scaled (numpy.ndarray): The scaled feature data.
            - y_one_hot (numpy.ndarray): The one-hot encoded labels.
            - le (LabelEncoder): The fitted label encoder.
            - scaler (StandardScaler): The fitted scaler.
    """
    # Separate features X (landmarks) and labels y (ASL letter)
    X = df.drop('label', axis=1).values  # Features: all columns except 'label'
    y = df['label'].values               # Labels: 'A', 'B', 'C', etc.
    
    # Initialize the LabelEncoder to convert categorical labels to numerical labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # 'A' -> 0, 'B' -> 1, 'C' -> 2, etc.
    
    # Apply One-Hot Encoding to the numerical labels for multi-class classification
    y_one_hot = tf.keras.utils.to_categorical(y_encoded)
    
    # Initialize the StandardScaler to standardize feature data
    scaler = StandardScaler()

    # Scale features to have zero mean and unit variance
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_one_hot, le, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets,
    maintaining the proportion of classes in both sets.

    Args:
        X (numpy.ndarray): The feature data.
        y (numpy.ndarray): The one-hot encoded labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple:
            - X_train (numpy.ndarray): Training features.
            - X_test (numpy.ndarray): Testing features.
            - y_train (numpy.ndarray): Training labels.
            - y_test (numpy.ndarray): Testing labels.
    """
    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Preserves class distribution in training and testing sets
    )
    
    # Print the number of samples in training and testing sets
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def add_noise(X, noise_factor=0.01):
    """
    Add Gaussian noise to data to help with generalization.
    
    Args:
        X (numpy.ndarray): Feature data to which noise will be added.
        noise_factor (float): Magnitude of the noise to be added.

    Returns:
        numpy.ndarray: Feature data with added noise.
    """
    noisy_X = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return noisy_X

def build_model(input_dim, num_classes):
    """
    Construct and compile the neural network model for multi-class classification.

    Args:
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: The compiled neural network model.
    """
    # Initialize a Sequential model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),    # First hidden layer with ReLU activation
        Dropout(0.5),                   # Dropout layer to prevent overfitting by randomly dropping 50% of neurons
        Dense(256, activation='relu'),  # Second hidden layer
        Dropout(0.4),                   # Dropout layer with 40% rate
        Dense(128, activation='relu'),  # Third hidden layer
        Dropout(0.3),                   # Dropout layer with 30% rate
        Dense(num_classes, activation='softmax')  # Output layer for multi-class probabilities
    ])
    
    # Compile the model optimizer, loss function, and metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',    # Suitable for multi-class classification
        metrics=['accuracy']                # Monitors accuracy during training
    )
    
    # Print summary of the model's architecture
    model.summary()
    
    return model

def plot_history(history):
    """
    Plot training & validation accuracy and loss over epochs.
    Helps diagnosies overfitting or underfitting by comparing training and valication metrics.

    Args:
        history (tensorflow.keras.callbacks.History): History object returned by model.fit().
    """
    # Create a figure with two subplots
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')           # Training accuracy over epochs
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Validation accuracy
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Training and validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')                   # Training loss over epochs
    plt.plot(history.history['val_loss'], label='Validation Loss')          # Validation loss
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Display plots
    plt.show()

def evaluate_model(model, X_test, y_test, le):
    """
    Evaluate the trained model on the test set and displays metrics.
    Loss, accuracy, classification report, and confusion matrix are evaluated.

    Args:
        model (tensorflow.keras.Model): The trained neural network model.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing labels (one-hot encoded).
        le (LabelEncoder): The label encoder to decode numerical labels back to original.
    """
    # Evaluate the model's performance on the test set (20% of dataset input)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate probability predictions for the test set
    y_pred_prob = model.predict(X_test)

    # Convert probability predictions to class labels
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Convert one-hot encoded true labels to class labels
    y_true = np.argmax(y_test, axis=1)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    # Generate confusion matrix to see mismatches
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def save_model_and_objects(model, le, scaler, model_path='StaticSignModel.keras',
                           le_path='StaticLabelEncoder.pkl', scaler_path='StaticStandardScaler.pkl'):
    """
    Saves the trained model and preprocessing objects.

    Args:
        model (tensorflow.keras.Model): The trained neural network model.
        le (StaticLabelEncoder): The fitted label encoder.
        scaler (StaticStandardScaler): The fitted scaler.
        model_path (str): File path to save the trained model.
        le_path (str): File path to save the label encoder.
        scaler_path (str): File path to save the scaler.
    """
    # Save the trained model as a .keras file
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Save the LabelEncoder
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved as {le_path}")
    
    # Save the StandardScaler using pickle to apply the same scaling during inference
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved as {scaler_path}")

def main():
    """
    The main function to execute the data loading, preprocessing, training, evaluation, and saving steps.
    """
    # Step 1: Load and inspect the dataset
    df = load_and_inspect_data('StaticSignData.csv')
    
    # Step 2: Preprocess data (encode labels and scale features)
    X_scaled, y_one_hot, le, scaler = preprocess_data(df)
    
    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_one_hot)

    # Step 3.5: Add noise to the data to add generality for training
    X_train_noisy = add_noise(X_train)
    
    # Step 4: Build the neural network model
    num_classes = y_one_hot.shape[1] # Determine the number of unique classes
    model = build_model(input_dim=X_train_noisy.shape[1], num_classes=num_classes)
    
    # Define callbacks for training
    early_stop = EarlyStopping(
        monitor='val_loss',         # Monitor validation loss
        patience=15,                # Stop training after 15 epochs without improvement
        restore_best_weights=True   # Restore model weights from the epoch with the best values
    )
    
    # Saves the model after each epoch if there is improvement in val_loss
    checkpoint = ModelCheckpoint(
        'StaticSignModel.keras',          # Path to save the model
        monitor='val_loss',         # Monitor validation loss
        save_best_only=True,        # Only save the best model
        verbose=1
    )
    
    # Compute class weights to handle class imbalance
    y_train_labels = np.argmax(y_train, axis=1)                                     # Convert one-hot to class labels
    class_weights_array = tf.keras.utils.to_categorical(y_train_labels).sum(axis=0) # Calculate class frequencies
    class_weights = {}
    total = len(y_train_labels)                                 # Training samples total
    for i in range(num_classes):
        count = np.sum(y_train_labels == i)                     # Number of samples in class i
        if count != 0:
            class_weights[i] = total / (num_classes * count)    # Inverse frequency weighting
        else:
            class_weights[i] = 1.0                              # Avoid division by zero
    
    # print computed weights
    print("Class Weights:", class_weights)
    
    # Step 5: Train the model with training data
    history = model.fit(
        X_train_noisy, 
        y_train,
        epochs=200,                         # Max number of epochs
        batch_size=32,                      # Number of samples per gradient update
        validation_split=0.2,               # 20% of training data for validation
        callbacks=[early_stop, checkpoint], # Callbacks to apply during training (shorten training)
        class_weight=class_weights          # Apply class weights
    )
    
    # Step 6: Plot training and validation history
    plot_history(history)
    
    # Evaluate trained model on test set
    evaluate_model(model, X_test, y_test, le)
    
    # Step 7: Save the model and preprocessing objects
    save_model_and_objects(model, le, scaler)

if __name__ == "__main__":
    # Run main function on program execution
    main()
