import os
import time
import numpy as np
import tensorflow as tf
from edge.features import extract_features, TARGET_SAMPLES
from streaming.encoder import ImaAdpcmEncoder, ImaAdpcmDecoder
from training.dataset import generate_synthetic_data

def run_benchmarks():
    print("====================================================")
    print("             EdgeWake Telemetry Benchmark           ")
    print("====================================================")
    
    # 1. Generate test signals
    print("Generating test signals...")
    duration = 5.0 # 5 seconds of audio
    sample_rate = 16000
    test_pcm = np.random.normal(0, 0.1, int(duration * sample_rate)).astype(np.int16)
    one_second_pcm = test_pcm[:TARGET_SAMPLES]
    
    # 2. Benchmark Feature Extraction
    print("Benchmarking Log-Mel Spectrogram extraction...")
    runs = 100
    start = time.perf_counter()
    for _ in range(runs):
        _ = extract_features(one_second_pcm)
    end = time.perf_counter()
    feat_latency = (end - start) / runs * 1000
    print(f"  Feature Extraction Latency (1.0s window): {feat_latency:.3f} ms")
    
    # 3. Benchmark ADPCM Compression
    print("Benchmarking IMA ADPCM Codec...")
    encoder = ImaAdpcmEncoder()
    decoder = ImaAdpcmDecoder()
    
    start = time.perf_counter()
    for _ in range(runs):
        encoded = encoder.encode(test_pcm)
    end = time.perf_counter()
    enc_latency = (end - start) / runs * 1000
    
    start = time.perf_counter()
    for _ in range(runs):
        _ = decoder.decode(encoded, len(test_pcm))
    end = time.perf_counter()
    dec_latency = (end - start) / runs * 1000
    
    pcm_bytes = len(test_pcm) * 2
    adpcm_bytes = len(encoded)
    comp_ratio = pcm_bytes / adpcm_bytes
    
    print(f"  Encode Latency (5.0s audio): {enc_latency:.3f} ms")
    print(f"  Decode Latency (5.0s audio): {dec_latency:.3f} ms")
    print(f"  Compression Ratio: {comp_ratio:.2f}x (PCM {pcm_bytes:,}B vs ADPCM {adpcm_bytes:,}B)")
    
    # 4. Inference Latencies
    tflite_model_path = "models/edgewake_int8.tflite"
    h5_model_path = "models/edgewake_model.h5"
    
    features = extract_features(one_second_pcm)
    features_batch = np.expand_dims(features, axis=(0, -1))
    
    if os.path.exists(h5_model_path):
        print("Benchmarking Float32 Keras model inference...")
        model = tf.keras.models.load_model(h5_model_path)
        # Warmup
        for _ in range(10):
            _ = model.predict(features_batch, verbose=0)
        start = time.perf_counter()
        for _ in range(50):
            _ = model.predict(features_batch, verbose=0)
        end = time.perf_counter()
        float_latency = (end - start) / 50 * 1000
        print(f"  Float32 Inference Latency: {float_latency:.2f} ms")
    else:
        print("  Float32 model not trained. Skipping float benchmark.")
        
    if os.path.exists(tflite_model_path):
        print("Benchmarking INT8 TFLite model inference...")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Format input data matching scale
        input_scale, input_zero_point = input_details[0]['quantization']
        if input_details[0]['dtype'] == np.int8:
            input_data = (features_batch / input_scale) + input_zero_point
            input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)
        else:
            input_data = features_batch.astype(np.float32)
            
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
        start = time.perf_counter()
        for _ in range(50):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        end = time.perf_counter()
        int8_latency = (end - start) / 50 * 1000
        print(f"  INT8 Quantized Inference Latency: {int8_latency:.2f} ms")
    else:
        print("  INT8 TFLite model not quantized. Skipping INT8 benchmark.")
        
    print("====================================================")

if __name__ == "__main__":
    run_benchmarks()
