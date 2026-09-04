import numpy as np

# Feature extraction parameters matching MCU targets
SAMPLE_RATE = 16000
FRAME_LEN = 400       # 25 ms
FRAME_STEP = 160      # 10 ms
NUM_MELS = 40
NUM_FRAMES = 100
NUM_FFT = 512
# Total samples needed for exactly 100 frames
TARGET_SAMPLES = (NUM_FRAMES - 1) * FRAME_STEP + FRAME_LEN  # 16240 samples (~1.015s)

def hz_to_mel(hz):
    """Converts a frequency in Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    """Converts a Mel scale value back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def compute_mel_filterbank(num_fft, num_mels, sample_rate, low_freq=0.0, high_freq=None):
    """Generates a Mel filterbank matrix."""
    if high_freq is None:
        high_freq = sample_rate / 2.0
    
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Map Hz frequencies to FFT bin indices
    bin_points = np.floor((num_fft + 1) * hz_points / sample_rate).astype(int)
    
    filters = np.zeros((num_fft // 2 + 1, num_mels), dtype=np.float32)
    for i in range(num_mels):
        # Left slope of the triangular filter
        for j in range(bin_points[i], bin_points[i+1]):
            filters[j, i] = (j - bin_points[i]) / (bin_points[i+1] - bin_points[i] + 1e-10)
        # Right slope of the triangular filter
        for j in range(bin_points[i+1], bin_points[i+2]):
            filters[j, i] = (bin_points[i+2] - j) / (bin_points[i+2] - bin_points[i+1] + 1e-10)
            
    return filters

# Precompute filterbank and window function to optimize CPU
MEL_FILTERBANK = compute_mel_filterbank(NUM_FFT, NUM_MELS, SAMPLE_RATE)
HAMMING_WINDOW = np.hamming(FRAME_LEN).astype(np.float32)

def extract_features(audio):
    """Computes a standardized Log-Mel Spectrogram of shape (NUM_FRAMES, NUM_MELS) from raw audio."""
    # Ensure standard length
    if len(audio) != TARGET_SAMPLES:
        if len(audio) > TARGET_SAMPLES:
            audio = audio[:TARGET_SAMPLES]
        else:
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode='constant')
            
    # Normalize input amplitude to float32 range [-1, 1] if integer
    if audio.dtype == np.int16 or np.max(np.abs(audio)) > 1.0:
        audio = audio.astype(np.float32) / 32768.0

    # Framing
    num_frames = (len(audio) - FRAME_LEN) // FRAME_STEP + 1
    shape = (num_frames, FRAME_LEN)
    strides = (audio.strides[0] * FRAME_STEP, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    
    # Apply Hamming Window
    windowed = frames * HAMMING_WINDOW
    
    # Compute Power FFT
    dft = np.fft.rfft(windowed, n=NUM_FFT, axis=-1)
    power_spectrogram = np.abs(dft) ** 2 / FRAME_LEN
    
    # Apply Mel filterbank
    mel_spectrogram = np.dot(power_spectrogram, MEL_FILTERBANK)
    
    # Log-Mel with stable floor
    log_mel = np.log(mel_spectrogram + 1e-10)
    
    # Standardized scaling to roughly [-2.0, 2.0] range
    norm_log_mel = (log_mel + 12.0) / 6.0
    return norm_log_mel.astype(np.float32)

