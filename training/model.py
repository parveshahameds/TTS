import tensorflow as tf
from tensorflow.keras import layers, models

def create_dscnn_model(input_shape=(100, 40, 1), num_classes=3):
    """Creates a tiny Depthwise-Separable CNN (DS-CNN) model optimized for microcontrollers."""
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=input_shape))
    
    # 1. Standard Conv2D Layer (acting as initial feature processing)
    # Using larger kernel in time dimension (10, 4) and stride (2, 2) to reduce dimensions quickly
    model.add(layers.Conv2D(16, kernel_size=(10, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # 2. DS-CNN Block 1
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # 3. DS-CNN Block 2
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # 4. DS-CNN Block 3
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Global Average Pooling to flatten spatial dimensions (MCU friendly compared to Flatten)
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense classification layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    model = create_dscnn_model()
    model.summary()
