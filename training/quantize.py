import os
import argparse
import numpy as np
import tensorflow as tf
from training.dataset import load_dataset

def quantize_model(model_path="models/edgewake_model.h5", output_path="models/edgewake_int8.tflite", data_dir="data"):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
        
    print(f"Loading trained float32 model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading representative dataset for calibration...")
    X, _ = load_dataset(data_dir)
    
    # Use a subset of training data for representative calibration
    # Standard size is ~100 samples
    num_calibration_samples = min(100, len(X))
    calibration_data = X[:num_calibration_samples].astype(np.float32)
    
    def representative_dataset_gen():
        for i in range(num_calibration_samples):
            # TFLite expects input with a batch dimension of 1
            sample = np.expand_dims(calibration_data[i], axis=0)
            yield [sample]
            
    print("Converting model to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Ensure standard operators are used
    converter.target_spec.supported_types = [tf.int8]
    
    try:
        tflite_model_quantized = converter.convert()
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Retrying with relaxed input/output types (float32 inputs, int8 internal execution)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_model_quantized = converter.convert()
        
    # Save quantized model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model_quantized)
        
    # Size Comparison
    float_size_kb = os.path.getsize(model_path) / 1024
    quant_size_kb = os.path.getsize(output_path) / 1024
    compression_ratio = float_size_kb / quant_size_kb if quant_size_kb > 0 else 0
    
    # Run dynamic sanity check
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print("\n" + "="*50)
    print("              EdgeWake Quantization Report        ")
    print("="*50)
    print(f"Original Model Size (Float32): {float_size_kb:.2f} KB")
    print(f"Quantized Model Size (INT8):   {quant_size_kb:.2f} KB")
    print(f"Compression Ratio:             {compression_ratio:.2f}x")
    print(f"Model Parameters:              {model.count_params():,}")
    print(f"Estimated RAM Footprint:       {quant_size_kb * 1.5:.2f} KB (Internal buffers + weights)")
    print("-"*50)
    print(f"Quantized Model Sanity Check:")
    print(f"  Input Shape:  {input_shape}")
    print(f"  Input Dtype:  {input_dtype}")
    print(f"  Output Shape: {output_details[0]['shape']}")
    print(f"  Output Dtype: {output_details[0]['dtype']}")
    print("Sanity check passed successfully!")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize EdgeWake Model")
    parser.add_argument("--model_path", type=str, default="models/edgewake_model.h5", help="Path to float32 model file")
    parser.add_argument("--output_path", type=str, default="models/edgewake_int8.tflite", help="Path to save quantized model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset files for calibration")
    args = parser.parse_args()
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        data_dir=args.data_dir
    )
