import numpy as np

class AcousticGate:
    def __init__(self, sample_rate=16000, frame_len=1600, noise_floor_alpha=0.98, speech_threshold_factor=2.5):
        """
        Cheap DSP VAD / Acoustic Gate.
        
        Args:
            sample_rate (int): Audio sample rate.
            frame_len (int): Length of analyzed audio chunk (1600 samples = 100ms).
            noise_floor_alpha (float): Smoothing factor for running noise floor estimation.
            speech_threshold_factor (float): Multiplier above noise floor to trigger gate.
        """
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.noise_floor_alpha = noise_floor_alpha
        self.speech_threshold_factor = speech_threshold_factor
        
        self.noise_floor_rms = 0.001 # Initialize with small value
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_limit = 20  # First 20 frames (2 seconds) used for calibration
        
    def get_rms(self, audio_chunk):
        """Computes root-mean-square energy of the audio chunk."""
        # Convert to float32 normalized if needed
        if audio_chunk.dtype == np.int16:
            chunk_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            chunk_float = audio_chunk.astype(np.float32)
            
        rms = np.sqrt(np.mean(chunk_float ** 2) + 1e-10)
        return rms
        
    def get_zcr(self, audio_chunk):
        """Computes Zero Crossing Rate (ZCR)."""
        # Count sign changes
        zero_crossings = np.sum(np.diff(np.sign(audio_chunk)) != 0)
        return zero_crossings / len(audio_chunk)
        
    def process_frame(self, audio_chunk):
        """
        Analyzes frame and returns True if speech-like activity is detected.
        Updates noise floor running estimates.
        """
        rms = self.get_rms(audio_chunk)
        zcr = self.get_zcr(audio_chunk)
        
        # Calibration phase
        if not self.is_calibrated:
            self.calibration_frames += 1
            if self.calibration_frames == 1:
                self.noise_floor_rms = rms
            else:
                self.noise_floor_rms = (self.noise_floor_rms * (self.calibration_frames - 1) + rms) / self.calibration_frames
                
            if self.calibration_frames >= self.calibration_limit:
                self.is_calibrated = True
                print(f"Acoustic VAD Calibrated. Ambient RMS Noise Floor: {self.noise_floor_rms:.5f}")
            return True, rms, zcr, self.noise_floor_rms # Active during calibration
            
        # Dynamically update noise floor if current RMS is lower than estimated noise floor
        # Or slowly drift towards current RMS when silent
        if rms < self.noise_floor_rms:
            self.noise_floor_rms = rms
        else:
            # Slow running average update
            self.noise_floor_rms = (self.noise_floor_alpha * self.noise_floor_rms) + ((1.0 - self.noise_floor_alpha) * rms)
            
        # Speech Gate conditions
        # Speech RMS typically has to be significantly above background noise floor.
        # Zero-Crossing Rate of speech is typically between 0.05 and 0.4.
        rms_active = rms > (self.noise_floor_rms * self.speech_threshold_factor)
        zcr_active = 0.03 < zcr < 0.5
        
        is_active = rms_active and zcr_active
        return is_active, rms, zcr, self.noise_floor_rms
