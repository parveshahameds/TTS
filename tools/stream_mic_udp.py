#!/usr/bin/env python3
"""
stream_mic_udp.py
-----------------
Captures real-time audio from the default microphone using PyAudio (16kHz, 16-bit PCM mono)
and streams raw binary PCM packets over UDP to an ESP32 or external receiver on port 5005.

Usage:
    python tools/stream_mic_udp.py --ip <ESP32_IP_ADDRESS> [--port 5005] [--chunk 512]
"""

import sys
import time
import socket
import argparse
import numpy as np
import pyaudio

# Audio Configuration
SAMPLE_RATE = 16000     # 16 kHz
CHANNELS = 1            # Mono
AUDIO_FORMAT = pyaudio.paInt16 # 16-bit Linear PCM
DEFAULT_CHUNK_SIZE = 512       # 512 samples per packet (1024 bytes, 32ms)
DEFAULT_PORT = 5005

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream 16kHz 16-bit Mono PCM audio from microphone to ESP32 over UDP."
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.4.1",
        help="Target IP address of the ESP32 (default: 192.168.4.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Target UDP port on ESP32 (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of audio samples per UDP packet (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional input device index (default: system default microphone)"
    )
    return parser.parse_args()

def stream_audio(ip: str, port: int, chunk_size: int, device_index: int = None):
    # Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Display available input devices if requested or for info
    try:
        default_device_info = p.get_default_input_device_info()
        device_name = default_device_info.get("name", "Default Microphone")
    except Exception:
        device_name = "System Default"

    print("=" * 60)
    print("🎙️  EdgeWake Micro-Streamer (PyAudio -> UDP)")
    print("=" * 60)
    print(f" • Target ESP32 IP   : {ip}")
    print(f" • Target UDP Port   : {port}")
    print(f" • Sampling Rate     : {SAMPLE_RATE} Hz")
    print(f" • Audio Format      : 16-bit PCM Mono (int16)")
    print(f" • Chunk Size        : {chunk_size} samples ({chunk_size * 2} bytes / packet)")
    print(f" • Audio Source      : {device_name}")
    print("=" * 60)
    print("Streaming live audio... Press Ctrl+C to stop.\n")

    # Open PyAudio Stream
    try:
        stream = p.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
    except Exception as e:
        print(f"❌ Failed to open audio input stream: {e}")
        p.terminate()
        sock.close()
        sys.exit(1)

    total_packets = 0
    total_bytes = 0
    start_time = time.time()
    last_stat_time = start_time

    try:
        while True:
            # Read raw PCM bytes from microphone
            # exception_on_overflow=False prevents crashes on transient OS audio lags
            raw_pcm = stream.read(chunk_size, exception_on_overflow=False)
            
            # Send raw PCM packet over UDP to ESP32
            sock.sendto(raw_pcm, (ip, port))
            
            total_packets += 1
            total_bytes += len(raw_pcm)

            # Calculate real-time RMS for visual volume meter
            now = time.time()
            if now - last_stat_time >= 0.2: # Update console every 200ms
                audio_array = np.frombuffer(raw_pcm, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)) if len(audio_array) > 0 else 0
                meter_len = int(min(30, (rms / 3000.0) * 30))
                vu_meter = "█" * meter_len + "░" * (30 - meter_len)
                
                elapsed = now - start_time
                kbps = (total_bytes * 8) / (elapsed * 1000) if elapsed > 0 else 0
                
                sys.stdout.write(
                    f"\r[{vu_meter}] Level: {rms:6.1f} | Sent: {total_packets:6d} pkts ({total_bytes / 1024:6.1f} KB) | Rate: {kbps:5.1f} kbps"
                )
                sys.stdout.flush()
                last_stat_time = now

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping audio streaming...")
    finally:
        # Clean shutdown
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        print("✅ PyAudio stream and UDP socket closed cleanly.")

if __name__ == "__main__":
    args = parse_args()
    stream_audio(
        ip=args.ip,
        port=args.port,
        chunk_size=args.chunk,
        device_index=args.device_index
    )
