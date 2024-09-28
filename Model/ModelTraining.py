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

def load_and_inspect_data(csv_path='sign_data.csv'):
    """
    Load the dataset and perform initial inspections.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    print("First five rows of the dataset:")
    print(df.head())
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Handle missing values if any
    if df.isnull().values.any():
        print("\nHandling missing values by dropping rows with missing data.")
        df = df.dropna()
    
    return df

def preprocess_data(df):
    """
    Encode labels and scale features.
    """
    # Separate features and labels
    X = df.drop('label', axis=1).values  # Features: landmarks
    y = df['label'].values               # Labels: 'A' or 'B'
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # 'A' -> 0, 'B' -> 1
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, le, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    """
    Build and compile the neural network model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

def plot_history(history):
    """
    Plot training & validation accuracy and loss.
    """
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

def evaluate_model(model, X_test, y_test, le):
    """
    Evaluate the model on the test set and display metrics.
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def save_model_and_objects(model, le, scaler, model_path='sign_model_A_B.h5',
                           le_path='label_encoder_A_B.pkl', scaler_path='scaler_A_B.pkl'):
    """
    Save the trained model and preprocessing objects.
    """
    # Save the trained model
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Save the LabelEncoder and StandardScaler
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved as {le_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved as {scaler_path}")

def main():
    # Step 1: Load and inspect data
    df = load_and_inspect_data('sign_data.csv')
    
    # Step 2: Preprocess data
    X_scaled, y_encoded, le, scaler = preprocess_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)
    
    # Step 4: Build the model
    model = build_model(input_dim=X_train.shape[1])
    
    # Step 5: Define callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'sign_model_A_B.keras', 
        monitor='val_loss', 
        save_best_only=True
    )
    
    # Step 6: Train the model
    history = model.fit(
        X_train, 
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,  # 20% of training data for validation
        callbacks=[early_stop, checkpoint]
    )
    
    # Step 7: Plot training history
    plot_history(history)
    
    # Step 8: Evaluate the model
    evaluate_model(model, X_test, y_test, le)
    
    # Step 9: Save the model and preprocessing objects
    save_model_and_objects(model, le, scaler)

if __name__ == "__main__":
    main()
