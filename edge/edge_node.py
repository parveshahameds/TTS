import time
import threading
import numpy as np
from edge.audio_stream import AudioMicrophoneStream
from edge.vad import AcousticGate
from edge.features import extract_features, TARGET_SAMPLES
from edge.kws import KWSInterpreter
from edge.temporal import TemporalVerifier
from edge.ring_buffer import AudioRingBuffer
from streaming.client import StreamingClient, COMPRESSION_NONE, COMPRESSION_ADPCM

class EdgeWakeNode:
    def __init__(self, host="127.0.0.1", port=5000, model_path="models/edgewake_int8.tflite", compression=COMPRESSION_NONE):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.compression = compression
        
        # Audio stream reads chunks of 160 samples (10ms)
        self.stream = AudioMicrophoneStream(block_size=160)
        
        # VAD operates on 100ms blocks (1600 samples)
        self.vad = AcousticGate(frame_len=1600, speech_threshold_factor=2.0)
        
        # KWS Interpreter loads INT8 model
        self.kws = KWSInterpreter(model_path=model_path)
        
        # Temporal smoothing
        self.temporal = TemporalVerifier(threshold=0.65, min_consecutive_frames=3, smoothing_factor=0.6)
        
        # Pre-roll ring buffer (500ms = 8000 samples)
        self.ring_buffer = AudioRingBuffer(capacity=8000)
        
        # TCP connection to server
        self.client = StreamingClient(host=host, port=port, compression=compression)
        
        # Buffers for sliding audio feature extraction
        self.audio_window = np.zeros(TARGET_SAMPLES, dtype=np.int16)
        
        self.is_running = False
        self.state = "LOCAL_LISTENING" # LOCAL_LISTENING or STREAMING
        self.streaming_duration = 5.0 # Max stream duration in seconds
        self.silence_timeout = 1.5 # Auto stop after 1.5s of silence
        self.state_start_time = 0.0
        self.last_speech_time = 0.0
        
        # Stats for dashboard consumption
        self.stats = {
            "is_active": False,
            "confidence": 0.0,
            "vad_status": "Idle",
            "cpu_saved": True,
            "rms_value": 0.0,
            "noise_floor": 0.0,
            "last_latency_ms": 0.0
        }
        
    def get_stats(self):
        """Returns live statistics for the GUI dashboard."""
        return self.stats
        
    def start(self):
        """Starts the edge execution node."""
        if not self.client.connect():
            print("ASR server must be running. Retrying network connection continuously in background...")
            
        self.stream.start()
        self.is_running = True
        
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.is_running = False
        self.stream.stop()
        self.client.disconnect()
        
    def _processing_loop(self):
        """Main processing thread."""
        accumulated_samples = []
        accumulated_samples = []
        
        # Estimate CPU tracking variables
        kws_infer_count = 0
        total_eval_periods = 0
        
        print("EdgeWake node processing loop started.")
        
        while self.is_running:
            # Read 10ms (160 samples) block from mic queue
            block = self.stream.read(block=True, timeout=0.1)
            if block is None:
                continue
                
            # Write to pre-roll ring buffer
            self.ring_buffer.write(block)
            
            # Slide audio window buffer
            self.audio_window = np.concatenate((self.audio_window[len(block):], block))
            
            # Accumulate samples for VAD frame (100ms / 1600 samples)
            accumulated_samples.append(block)
            if len(accumulated_samples) < 10:
                # If we are streaming, send audio blocks immediately to minimize latency
                if self.state == "STREAMING":
                    self.client.send_audio(block, timestamp=time.time())
                continue
                
            # We have accumulated 100ms of audio
            vad_chunk = np.concatenate(accumulated_samples)
            accumulated_samples = []
            total_eval_periods += 1
            
            # 1. Run VAD (Acoustic Gate)
            is_speech, rms, zcr, noise_floor = self.vad.process_frame(vad_chunk)
            
            self.stats["rms_value"] = float(rms)
            self.stats["noise_floor"] = float(noise_floor)
            self.stats["vad_status"] = "Speech" if is_speech else "Silence"
            
            if self.state == "LOCAL_LISTENING":
                # Adaptive computation: Only run KWS inference if VAD detects speech
                if is_speech:
                    self.stats["cpu_saved"] = False
                    kws_infer_count += 1
                    
                    # 2. Extract Log-Mel features
                    features = extract_features(self.audio_window)
                    
                    # 3. KWS model inference
                    probs = self.kws.predict(features)
                    keyword_prob = float(probs[0])
                    self.stats["confidence"] = keyword_prob
                    
                    # 4. Temporal verification
                    is_triggered, smoothed_prob = self.temporal.process_probability(keyword_prob)
                    
                    if is_triggered:
                        trigger_time = time.time()
                        print(f"\nWake Word DETECTED! Conf: {smoothed_prob:.2f}. Switching to Streaming...")
                        
                        # Fetch pre-roll (500ms)
                        pre_roll = self.ring_buffer.get_content()
                        
                        # Begin persistent socket stream
                        self.client.start_stream(pre_roll, trigger_time)
                        
                        self.state = "STREAMING"
                        self.state_start_time = trigger_time
                        self.last_speech_time = trigger_time
                        self.stats["is_active"] = True
                else:
                    self.stats["cpu_saved"] = True
                    # Reset temporal verifier slightly on silence
                    _, _ = self.temporal.process_probability(0.0)
                    self.stats["confidence"] = 0.0
                    
            elif self.state == "STREAMING":
                # Check VAD for auto-stop conditions
                current_time = time.time()
                
                # Check speech activity to reset silence timeout
                if is_speech:
                    self.last_speech_time = current_time
                    
                # Auto-stop conditions:
                # 1. Silence duration exceeds timeout (e.g. 1.5s)
                # 2. Max stream limit reached (e.g. 5s)
                silence_elapsed = current_time - self.last_speech_time
                total_elapsed = current_time - self.state_start_time
                
                if silence_elapsed >= self.silence_timeout or total_elapsed >= self.streaming_duration:
                    print("Silence or timeout threshold reached. Stopping stream...")
                    self.client.stop_stream(timestamp=current_time)
                    self.state = "LOCAL_LISTENING"
                    self.stats["is_active"] = False
                    self.ring_buffer.clear()
                    self.temporal.reset()
                    
            # Log CPU savings ratio
            cpu_idle_saving = (1.0 - (kws_infer_count / total_eval_periods)) * 100
            # Keep CPU saving logs clean
            if total_eval_periods % 100 == 0:
                # print(f"Idle CPU KWS Bypass Ratio: {cpu_idle_saving:.1f}%")
                pass

if __name__ == "__main__":
    # Test execution
    node = EdgeWakeNode()
    node.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        node.stop()
