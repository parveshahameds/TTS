import numpy as np

def pad_or_crop(audio, target_length=16000):
    """Pads or crops audio to a target length (16000 samples for 1 second)."""
    if len(audio) > target_length:
        # Random crop
        start = np.random.randint(0, len(audio) - target_length)
        return audio[start:start+target_length]
    elif len(audio) < target_length:
        # Pad with zeros
        pad_len = target_length - len(audio)
        pad_left = np.random.randint(0, pad_len + 1)
        pad_right = pad_len - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')
    return audio

def random_gain(audio, min_gain=0.5, max_gain=1.5):
    """Applies a random gain multiplier to the audio waveform."""
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain

def time_shift(audio, max_shift_ms=100, sample_rate=16000):
    """Shifts the audio waveform in time, wrapping or padding with zeros."""
    max_shift = int(max_shift_ms * sample_rate / 1000)
    if max_shift <= 0:
        return audio
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.pad(audio, (shift, 0), mode='constant')[:-shift]
    elif shift < 0:
        return np.pad(audio, (0, -shift), mode='constant')[-shift:]
    return audio

def mix_noise(audio, noise_samples, snr_db_range=(5, 15)):
    """Mixes background noise into the audio signal with a random SNR (in dB)."""
    if not noise_samples:
        return audio
    
    # Pick a random noise file
    noise = noise_samples[np.random.randint(0, len(noise_samples))]
    if len(noise) == 0:
        return audio
        
    # Crop a chunk of noise matching the audio length
    if len(noise) > len(audio):
        start = np.random.randint(0, len(noise) - len(audio))
        noise_chunk = noise[start:start+len(audio)]
    else:
        noise_chunk = np.pad(noise, (0, len(audio) - len(noise)), mode='wrap')
        
    # Standard SNR math
    snr_db = np.random.uniform(*snr_db_range)
    p_signal = np.mean(audio ** 2) + 1e-10
    p_noise = np.mean(noise_chunk ** 2) + 1e-10
    
    # scaling factor for noise
    k = np.sqrt(p_signal / (p_noise * (10 ** (snr_db / 10.0))))
    return audio + k * noise_chunk

def time_stretch(audio, stretch_range=(0.95, 1.05)):
    """Applies simple linear interpolation time-stretching to the audio."""
    stretch = np.random.uniform(*stretch_range)
    if stretch == 1.0:
        return audio
    num_samples = int(len(audio) / stretch)
    x = np.linspace(0, len(audio) - 1, num_samples)
    xp = np.arange(len(audio))
    stretched = np.interp(x, xp, audio)
    return pad_or_crop(stretched, len(audio))

def augment_waveform(audio, noise_samples=None):
    """Runs a complete random pipeline of audio augmentations."""
    # Ensure exactly 1 second first
    x = pad_or_crop(audio, 16000)
    
    # 80% chance to shift in time
    if np.random.random() < 0.8:
        x = time_shift(x, max_shift_ms=80)
        
    # 50% chance to stretch/compress time
    if np.random.random() < 0.5:
        x = time_stretch(x, stretch_range=(0.9, 1.1))
        
    # 80% chance to adjust volume
    if np.random.random() < 0.8:
        x = random_gain(x, min_gain=0.6, max_gain=1.4)
        
    # 60% chance to mix background noise if noise is available
    if noise_samples and np.random.random() < 0.6:
        x = mix_noise(x, noise_samples, snr_db_range=(7, 20))
        
    return x
