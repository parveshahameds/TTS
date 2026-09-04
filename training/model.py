import tensorflow as tf
from tensorflow.keras import layers, models

def create_dscnn_model(input_shape=(100, 40, 1), num_classes=3):
    """
    Creates a Depthwise-Separable CNN (DS-CNN) with 1D temporal phoneme modeling.
    Preserves sequential timing so syllables must occur in exact chronological order.
    """
    inp = layers.Input(shape=input_shape)
    
    # 1. Initial 2D Convolution (extracts time-frequency spectral features)
    x = layers.Conv2D(32, kernel_size=(5, 4), strides=(2, 2), padding='same', activation='relu')(inp)
    x = layers.Dropout(0.1)(x)
    
    # 2. DS-CNN Block 1
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(48, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # 3. DS-CNN Block 2 (temporal & spectral downsampling)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    # 4. DS-CNN Block 3 (temporal & spectral downsampling)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # 5. Pool across frequency dimension only, preserving the 13 temporal frames:
    # Shape transitions from (batch, 13, 5, 64) -> (batch, 13, 1, 64) -> (batch, 13, 64)
    x = layers.AveragePooling2D(pool_size=(1, 5))(x)
    x = layers.Reshape((13, 64))(x)
    
    # 6. 1D Temporal Convolution (models sequential syllable transitions)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # 7. Dense Classification Head
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inp, outputs=out, name="EdgeWake_KWS_DSCNN")
    return model

if __name__ == "__main__":
    model = create_dscnn_model()
    model.summary()

