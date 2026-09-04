import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from training.dataset import load_dataset
from training.model import create_dscnn_model

def train_model(data_dir="data", models_dir="models", epochs=40, batch_size=32):
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train split: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Val split: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # Create model
    input_shape = X.shape[1:]
    num_classes = len(np.unique(y))
    print(f"Creating DS-CNN model with input shape {input_shape} and {num_classes} classes...")
    model = create_dscnn_model(input_shape=input_shape, num_classes=num_classes)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_path = os.path.join(models_dir, "edgewake_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Training completed. Best model saved to: {model_path}")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EdgeWake KWS Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train WAV files")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

