# EdgeWake

> **"Listen locally. Wake instantly. Send only when needed."**

EdgeWake is a privacy-preserving, ultra-low-latency edge-to-cloud voice activation framework. It runs continuous, local audio capture and wake-word detection on a low-power edge processor (like an ESP32). Once the custom keyword is detected, it streams subsequent command audio via a persistent socket connection using compressed ADPCM encoding to a remote Automated Speech Recognition (ASR) server.

---

## 1. Problem & Context

### Why Cloud-Only Voice Activation Fails
1. **Privacy Violations**: Streaming raw, continuous microphone audio to cloud servers compromises privacy.
2. **Network Overhead**: Constantly transmitting 256kbps PCM audio exhausts bandwidth and battery.
3. **Server Costs**: Processing continuous silence/noise via heavyweight ASR models is computationally expensive.
4. **Wake Latency**: Initializing connections *after* keyword detection creates network handshake delays, causing the cloud server to miss the start of user instructions.

### The EdgeWake Solution
EdgeWake solves these problems through an optimized edge-cloud split:
- **Edge (Local)**: Runs a cheap DSP-based Acoustic Gate and a highly quantized INT8 Depthwise-Separable CNN (DS-CNN) wake-word detector (<32KB RAM, <10% idle CPU utilization).
- **Cloud (Remote)**: Remains idle until a verified wake event is received, then transcribes subsequent audio.

---

## 2. Architecture & Pipeline

```
Microphone ➔ Continuous Ring Buffer (500ms) 
                 │
                 ▼
          Acoustic Gate (VAD: RMS + ZCR) ──[Silence]──➔ Sleep Mode (Skip Inference)
                 │
            [Speech Active]
                 ▼
         Feature Extractor (Log-Mel Spectrogram)
                 │
                 ▼
         INT8 DS-CNN Model ➔ Temporal Smoothing ──[No Trigger]──➔ Continue Listening
                 │
             [Trigger]
                 ▼
      IMA ADPCM Compression (4:1) ➔ Persistent TCP Socket ➔ ASR Server ➔ Transcript
```

### Modular Pipeline Components
1. **Continuous Local Audio Capture**: Non-blocking microphone capture (16kHz, mono, 16-bit PCM).
2. **Acoustic Gate (VAD)**: A lightweight RMS and Zero-Crossing Rate (ZCR) gate. In silence, KWS neural evaluation is bypassed.
3. **Online Feature Extractor**: Computes a 40-band Log-Mel spectrogram window (25ms window, 10ms step, 1-second context).
4. **Tiny custom KWS Model**: A 3-class Depthwise-Separable CNN (DS-CNN) classifier (Classes: `KEYWORD`, `UNKNOWN_SPEECH`, `BACKGROUND`).
5. **Temporal Verification**: Integrates exponential smoothing and consecutive frame confirmation to reduce false activations.
6. **Pre-roll Ring Buffer**: Retains 500ms of audio locally to prepend to the stream, ensuring the start of commands isn't lost.
7. **Transport Layer**: A persistent TCP connection with optional 4-bit IMA ADPCM encoding (4:1 compression).
8. **ASR Server**: A modular transcription engine returning the final text.

---

## 3. TinyML Model & Quantization

### Model Architecture (DS-CNN)
Depthwise-Separable Convolutions (DS-CNN) reduce weights and MACs (Multiply-Accumulate operations) by splitting standard 2D convolutions into a spatial filter (Depthwise) and a channel filter (Pointwise):
- **Conv2D**: Stride (2,2) with Kernel (10,4) for fast dimension reduction.
- **DS-Conv Blocks**: Three cascaded layers with batch normalization, ReLU activation, and dropout.
- **Global Average Pooling**: Replaces dense flattening to reduce memory allocation.
- **Dense Output**: Softmax layer outputting class probabilities.

### Quantization (INT8 TFLite)
We use TensorFlow Lite Post-Training Quantization (PTQ) to convert all float32 weights, activations, and input/output layers to 8-bit integers:
- **Calibration**: Done using a representative dataset of Log-Mel features.
- **Inference Types**: Restructured to expect `int8` inputs and outputs (`[-128, 127]`).
- **Memory Footprint**: Fits within **32 KB RAM**, making it fully compatible with ESP32-S3 platforms using TensorFlow Lite Micro.

---

## 4. Running the Project

### Installation
Ensure you have a Python 3.9+ environment. Set up the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Record a Custom Dataset
Collect positive ("Hey Nova"), negative (similar phrases), and background (typing, noise) samples:
```bash
python3 tools/record_dataset.py
```

### Step 2: Train the Model
Train the DS-CNN model on the collected data (or synthetic data if folders are empty):
```bash
python3 training/train.py --epochs 30
```

### Step 3: Run Evaluation & Quantization
Assess metrics (F1, Precision, Recall, Latency) and convert to INT8 TFLite:
```bash
python3 training/evaluate.py
python3 training/quantize.py
```

### Step 4: Run Telemetry Benchmarks
Test feature extraction and TFLite model speeds locally:
```bash
python3 tools/benchmark.py
```

### Step 5: Launch the Interactive Dashboard & Server
The dashboard launches and connects to the ASR server automatically. To start:
```bash
streamlit run demo/app.py
```
1. In the sidebar, if you have not trained the model yet, click **"Setup Synthetic Model"** to compile a dummy model instantly for testing.
2. Toggle the **"Start EdgeWake Engine"** switch.
3. Speak **"Hey Nova"** into your microphone, then say a command (e.g., *"turn on the living room lights"*).
4. View the live privacy flowchart transition, check network bandwidths, and inspect the latency diagnostics.

### Running Tests
Execute unit tests to verify system stability:
```bash
pytest tests/test_components.py
```

---

## 5. Latency Telemetry

EdgeWake measures latency checkpoints down to the millisecond:
- **$T_0$**: Keyword Acoustic End (timestamp of the final acoustic frame matching the keyword).
- **$T_1$**: KWS Local Trigger Decision.
- **$T_2$**: Network Stream Start.
- **$T_3$**: Server receives first packet.
- **$T_4$**: Server starts ASR processing.
- **$T_5$**: Transcript generation complete.

The primary dashboard metric is **Keyword-End to Server Receive Latency ($T_3 - T_0$)**, which is optimized to `<100ms` thanks to persistent socket readiness and compressed ADPCM streaming.

---

## 6. ESP32-S3 Hardware Roadmap

To port this laptop prototype to an ESP32-S3 microcontroller:
1. **DMA I2S Acquisition**: Configure an I2S microphone (e.g., INMP441) using the ESP-IDF I2S driver to populate circular DMA memory buffers.
2. **DSP Feature Extraction**: Replace the NumPy-based Log-Mel Spectrogram with the ESP-DSP library (fast FFT implementation) and custom C-code for Mel-scale filtering.
3. **TFLite Micro Integration**: Load `models/edgewake_int8.tflite` into ESP32 flash. Run the TensorFlow Lite Micro library to load tensors and call the ESP-NN hardware-accelerated kernels.
4. **ADPCM Encoding**: Implement a 4-bit IMA ADPCM encoder in C to pack raw 16-bit samples into 4-bit bytes before transmission.
5. **Wi-Fi Transport**: Connect via BSD Sockets (`lwIP` stack) to the persistent TCP endpoint.
