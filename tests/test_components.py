import numpy as np
import pytest
from edge.features import extract_features, TARGET_SAMPLES
from edge.vad import AcousticGate
from edge.ring_buffer import AudioRingBuffer
from streaming.encoder import ImaAdpcmEncoder, ImaAdpcmDecoder
from streaming.protocol import TYPE_START, COMPRESSION_ADPCM, create_packet, parse_header

def test_feature_extraction_shape():
    # 1.015 seconds audio at 16000Hz is 16240 samples
    dummy_audio = np.random.normal(0, 0.1, TARGET_SAMPLES).astype(np.float32)
    features = extract_features(dummy_audio)
    assert features.shape == (100, 40)

def test_acoustic_gate_vad():
    gate = AcousticGate()
    # Dummy chunk (1600 samples)
    silent_chunk = np.zeros(1600, dtype=np.int16)
    rms = gate.get_rms(silent_chunk)
    assert rms == pytest.approx(0.0, abs=1e-5)
    
    noise_chunk = np.random.normal(0, 1000, 1600).astype(np.int16)
    rms_noise = gate.get_rms(noise_chunk)
    assert rms_noise > 0.0

def test_audio_ring_buffer():
    # Capacity 100 samples
    buffer = AudioRingBuffer(capacity=100)
    data1 = np.ones(60, dtype=np.int16)
    data2 = np.ones(50, dtype=np.int16) * 2
    
    buffer.write(data1)
    content = buffer.get_content()
    assert len(content) == 60
    assert np.all(content == 1)
    
    buffer.write(data2)
    content = buffer.get_content()
    assert len(content) == 100
    # First 50 elements should be from data1 (1s), second 50 from data2 (2s)
    assert np.all(content[:50] == 1)
    assert np.all(content[50:] == 2)

def test_adpcm_codec():
    encoder = ImaAdpcmEncoder()
    decoder = ImaAdpcmDecoder()
    
    # Generate simple sine wave
    t = np.linspace(0, 0.1, 1600)
    pcm = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    
    encoded = encoder.encode(pcm)
    assert len(encoded) == 800 # 4-bit packing (4:1 compression from 3200 bytes)
    
    decoded = decoder.decode(encoded, len(pcm))
    assert len(decoded) == len(pcm)
    
    # Check that reconstructed signal is highly correlated with original
    # (since ADPCM is lossy, we expect some noise but high similarity)
    correlation = np.corrcoef(pcm, decoded)[0, 1]
    assert correlation > 0.95

def test_temporal_verifier_margin_and_rejection():
    from edge.temporal import TemporalVerifier
    
    verifier = TemporalVerifier(threshold=0.80, min_consecutive_frames=3, margin=0.20)
    
    # 1. Test rejection on noise / background (e.g. Class 2 high)
    is_trig, conf = verifier.process_probability([0.3, 0.1, 0.6])
    assert not is_trig
    assert verifier.consecutive_count == 0
    
    # 2. Test rejection on unknown speech (Class 1 high)
    is_trig, conf = verifier.process_probability([0.45, 0.50, 0.05])
    assert not is_trig
    assert verifier.consecutive_count == 0
    
    # 3. Test rejection when margin is insufficient (e.g. 0.82 Keyword vs 0.75 Unknown)
    is_trig, conf = verifier.process_probability([0.82, 0.75, 0.05])
    assert not is_trig
    assert verifier.consecutive_count == 0
    
    # 4. Test trigger on 3 consecutive strong frames with proper margin
    strong_frame = [0.95, 0.03, 0.02]
    t1, _ = verifier.process_probability(strong_frame)
    assert not t1
    assert verifier.consecutive_count == 1
    
    t2, _ = verifier.process_probability(strong_frame)
    assert not t2
    assert verifier.consecutive_count == 2
    
    t3, _ = verifier.process_probability(strong_frame)
    assert t3 # Verified & Triggered on 3rd frame!

def test_kws_int8_interpreter():
    import os
    from edge.kws import KWSInterpreter
    
    model_path = "models/edgewake_int8.tflite"
    if os.path.exists(model_path):
        interpreter = KWSInterpreter(model_path)
        dummy_spec = np.zeros((100, 40), dtype=np.float32)
        probs = interpreter.predict(dummy_spec)
        assert len(probs) == 3
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

def test_server_client_lifecycle():
    import time
    from server.server import EdgeWakeASRServer
    from streaming.client import StreamingClient
    
    server = EdgeWakeASRServer(port=5055)
    server.start()
    time.sleep(0.1)
    
    client = StreamingClient(port=5055)
    connected = client.connect()
    assert connected
    
    # Send pre-roll and dummy chunk
    client.start_stream(np.zeros(800, dtype=np.int16), time.time())
    client.send_audio(np.zeros(160, dtype=np.int16))
    time.sleep(0.1)
    client.stop_stream()
    time.sleep(0.1)
    
    client.disconnect()
    server.stop()


