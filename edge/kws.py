import os
import numpy as np
import tensorflow as tf

class KWSInterpreter:
    def __init__(self, model_path="models/edgewake_int8.tflite"):
        """
        KWS Interpreter utilizing TFLite INT8 Quantized model.
        
        Args:
            model_path (str): Path to TFLite quantized model.
        """
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Quantized model not found at {model_path}. Train and quantize model first.")
            
        print(f"Loading TFLite model: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Scale and Zero points for quantization / dequantization
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
        
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
    def predict(self, spectrogram):
        """
        Runs inference on a single log-mel spectrogram window of shape (100, 40).
        
        Args:
            spectrogram (np.ndarray): Shape (100, 40) float32 Log-Mel Spectrogram.
        Returns:
            probabilities (np.ndarray): Shape (3,) probabilities of classes.
        """
        # Ensure it has channel dimension -> (1, 100, 40, 1)
        if len(spectrogram.shape) == 2:
            input_data = np.expand_dims(spectrogram, axis=(0, -1))
        elif len(spectrogram.shape) == 3:
            input_data = np.expand_dims(spectrogram, axis=0)
        else:
            input_data = spectrogram
            
        # Quantize float32 features to INT8 if the model expects it
        if self.input_dtype == np.int8:
            if self.input_scale != 0.0:
                input_data = (input_data / self.input_scale) + self.input_zero_point
                input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)
            else:
                input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
            
        # Set Tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run Inference
        self.interpreter.invoke()
        
        # Get Output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output back to float32 probabilities
        if self.output_details[0]['dtype'] == np.int8:
            if self.output_scale != 0.0:
                probs = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
            else:
                probs = output_data.astype(np.float32)
        else:
            probs = output_data
            
        # Apply Softmax if not already applied by model (TFLite output is 0.0-1.0 if softmax dense layer used)
        # We ensure it sums to 1.0
        probs = probs[0]
        # Clip to ensure valid probabilities
        probs = np.clip(probs, 0.0, 1.0)
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs /= prob_sum
            
        return probs
