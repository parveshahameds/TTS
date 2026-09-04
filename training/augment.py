import numpy as np

def align_speech_window(audio, target_length=16240, jitter_ms=30, sample_rate=16000):
    """
    Finds the speech active region using energy envelope and extracts a target_length window
    centered on the speech, with optional random temporal jitter.
    """
    if len(audio) <= target_length:
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')
        return audio
        
    frame_len = int(0.025 * sample_rate)  # 400
    hop = int(0.010 * sample_rate)        # 160
    num_frames = (len(audio) - frame_len) // hop + 1
    
    energies = np.array([
        np.sum(audio[i*hop : i*hop + frame_len]**2) 
        for i in range(num_frames)
    ])
    
    # Smooth energy
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(energies, kernel, mode='same')
    total = np.sum(smoothed)
    
    if total > 1e-6:
        center_frame = int(np.sum(np.arange(len(smoothed)) * smoothed) / total)
        center_sample = center_frame * hop + (frame_len // 2)
    else:
        center_sample = len(audio) // 2
        
    start = center_sample - (target_length // 2)
    if jitter_ms > 0:
        max_jitter = int(jitter_ms * sample_rate / 1000)
        start += np.random.randint(-max_jitter, max_jitter + 1)
        
    start = max(0, min(len(audio) - target_length, start))
    return audio[start : start + target_length]

def pad_or_crop(audio, target_length=16240):
    """Pads or crops audio to target length."""
    if len(audio) > target_length:
        return align_speech_window(audio, target_length, jitter_ms=0)
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), mode='constant')
    return audio

def random_gain(audio, min_gain=0.6, max_gain=1.4):
    """Applies a random gain multiplier."""
    return audio * np.random.uniform(min_gain, max_gain)

def mix_noise(audio, noise_samples, snr_db_range=(8, 25)):
    """Mixes background noise into the audio signal with a random SNR."""
    if not noise_samples:
        return audio
    noise = noise_samples[np.random.randint(0, len(noise_samples))]
    if len(noise) == 0:
        return audio
        
    if len(noise) > len(audio):
        start = np.random.randint(0, len(noise) - len(audio))
        noise_chunk = noise[start:start+len(audio)]
    else:
        noise_chunk = np.pad(noise, (0, len(audio) - len(noise)), mode='wrap')
        
    snr_db = np.random.uniform(*snr_db_range)
    p_signal = np.mean(audio ** 2) + 1e-10
    p_noise = np.mean(noise_chunk ** 2) + 1e-10
    k = np.sqrt(p_signal / (p_noise * (10 ** (snr_db / 10.0))))
    return audio + k * noise_chunk

def augment_waveform(audio, noise_samples=None, jitter_ms=30):
    """Runs audio augmentations with energy alignment, gain perturbation, and noise injection."""
    x = align_speech_window(audio, 16240, jitter_ms=jitter_ms)
    x = random_gain(x, min_gain=0.7, max_gain=1.3)
    if noise_samples and np.random.random() < 0.6:
        x = mix_noise(x, noise_samples, snr_db_range=(8, 25))
    return x

