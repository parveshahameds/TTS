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

def test_protocol_packets():
    payload = b"TESTPAYLOAD"
    timestamp = 1693400000.123
    
    packet = create_packet(TYPE_START, COMPRESSION_ADPCM, payload, seq_num=42, timestamp=timestamp)
    header = parse_header(packet[:20])
    
    assert header is not None
    assert header["packet_type"] == TYPE_START
    assert header["compression_type"] == COMPRESSION_ADPCM
    assert header["seq_num"] == 42
    assert abs(header["timestamp"] - timestamp) < 0.002 # Float precision within 1ms
    assert header["payload_len"] == len(payload)
