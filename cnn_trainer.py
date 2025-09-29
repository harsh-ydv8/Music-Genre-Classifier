import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,               
    MaxPooling1D,
    BatchNormalization,   
    Dropout,           
    Flatten,          
    Dense          
)

CSV_PATH = "features.csv"

features_df = pd.read_csv(CSV_PATH)

X = features_df.drop('genre_label', axis=1)
y = features_df['genre_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Reshaping data for CNN model ---")
X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}")
print(f"X_test_cnn shape: {X_test_cnn.shape}")

model = Sequential()
print("Sequential model canvas created successfully.")
model.add(Conv1D(filters=32,kernel_size=3,activation='relu',input_shape=(X_train_cnn.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print("\n--- Starting Model Training ---")
history = model.fit(
    X_train_cnn, y_train,         
    epochs=50,                     
    batch_size=32,                 
    validation_split=0.2 
)


print("\n--- Plotting Training and Validation History ---")

import matplotlib.pyplot as plt

def plot_history(history):
    """Plots accuracy and loss for training and validation sets."""
    
    # Create a figure with two subplots: one for accuracy, one for loss
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # --- Plot Training & Validation Accuracy ---
    # Access the accuracy history from the history object
    axs[0].plot(history.history["accuracy"], label="Training Accuracy")
    # Access the validation accuracy history
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend(loc="lower right")

    # --- Plot Training & Validation Loss ---
    # Access the loss history
    axs[1].plot(history.history["loss"], label="Training Loss")
    # Access the validation loss history
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend(loc="upper right")

    # Ensure the plots are nicely arranged
    plt.tight_layout()

    # Display the plots
    plt.show()

# Call the function with our history object
plot_history(history)

print("\n--- Saving the trained CNN model to disk ---")
model.save("music_genre_cnn.h5")
