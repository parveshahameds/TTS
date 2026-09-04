import os
import glob
import numpy as np
import soundfile as sf
from edge.features import extract_features, TARGET_SAMPLES
from training.augment import augment_waveform, align_speech_window, random_gain, mix_noise
from training.synthetic_speech import (
    generate_synthetic_phoneme_speech, 
    generate_synthetic_confuser, 
    generate_synthetic_noise_transient
)

# Labels
LABEL_KEYWORD = 0       # "Hey Nova"
LABEL_UNKNOWN = 1       # Other speech
LABEL_BACKGROUND = 2    # Silence / environmental noise
NUM_CLASSES = 3

def load_wav(filepath):
    """Loads a WAV file normalized to float32 [-1.0, 1.0]."""
    try:
        data, samplerate = sf.read(filepath)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def generate_synthetic_data(num_samples=90):
    """Generates synthetic audio data for testing when no recordings exist."""
    print("Generating balanced synthetic dataset...")
    X = []
    y = []
    
    for _ in range(num_samples // 3):
        # Keyword synthetic
        t = np.linspace(0, 1.0, TARGET_SAMPLES)
        kw = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        X.append(extract_features(kw))
        y.append(LABEL_KEYWORD)
        
        # Unknown synthetic
        unk = generate_synthetic_phoneme_speech()
        X.append(extract_features(unk))
        y.append(LABEL_UNKNOWN)
        
        # Background synthetic
        bg = generate_synthetic_noise_transient()
        X.append(extract_features(bg))
        y.append(LABEL_BACKGROUND)
        
    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    return X, y

def load_dataset(data_dir="data"):
    """Loads all WAV files from the dataset directory and computes log-mel spectrogram features."""
    pos_dir = os.path.join(data_dir, "positive")
    neg_dir = os.path.join(data_dir, "negative")
    bg_dir = os.path.join(data_dir, "background")
    
    pos_files = glob.glob(os.path.join(pos_dir, "*.wav"))
    neg_files = glob.glob(os.path.join(neg_dir, "*.wav"))
    bg_files = glob.glob(os.path.join(bg_dir, "*.wav"))
    
    print(f"Found {len(pos_files)} positive, {len(neg_files)} negative, and {len(bg_files)} background files.")
    
    if len(pos_files) == 0 and len(neg_files) == 0:
        return generate_synthetic_data()
        
    bg_noises = []
    for f in bg_files:
        audio = load_wav(f)
        if audio is not None:
            bg_noises.append(audio)
            
    X = []
    y = []
    
    # 1. Process Positive Samples (Keyword: Class 0)
    for f in pos_files:
        audio = load_wav(f)
        if audio is None:
            continue
        # Original aligned
        aligned = align_speech_window(audio, TARGET_SAMPLES, jitter_ms=0)
        X.append(extract_features(aligned))
        y.append(LABEL_KEYWORD)
        
        # Augment with temporal jitter, volume variations, and background noise mixing
        for _ in range(12):
            aug_audio = augment_waveform(audio, bg_noises, jitter_ms=35)
            X.append(extract_features(aug_audio))
            y.append(LABEL_KEYWORD)
            
    # 2. Process Negative Samples (Unknown Speech: Class 1)
    for f in neg_files:
        audio = load_wav(f)
        if audio is None:
            continue
        aligned = align_speech_window(audio, TARGET_SAMPLES, jitter_ms=0)
        X.append(extract_features(aligned))
        y.append(LABEL_UNKNOWN)
        
        for _ in range(15):
            aug_audio = augment_waveform(audio, bg_noises, jitter_ms=45)
            X.append(extract_features(aug_audio))
            y.append(LABEL_UNKNOWN)
            
    # Add diverse synthetic speech phonemes and confusers to the Unknown Speech class
    num_synthetic_speech = max(100, len(X) // 3)
    for _ in range(num_synthetic_speech):
        syn_speech = generate_synthetic_phoneme_speech()
        if bg_noises and np.random.random() < 0.4:
            syn_speech = mix_noise(syn_speech, bg_noises, snr_db_range=(10, 25))
        X.append(extract_features(syn_speech))
        y.append(LABEL_UNKNOWN)
        
    # Add phoneme confuser words ("Hey", "Nova", "Hello", "Never", etc.)
    for _ in range(50):
        confuser = generate_synthetic_confuser()
        X.append(extract_features(confuser))
        y.append(LABEL_UNKNOWN)

    # 3. Process Background Samples (Silence & Noise: Class 2)
    for bg in bg_noises:
        step = TARGET_SAMPLES // 2
        for start in range(0, len(bg) - TARGET_SAMPLES, step):
            chunk = bg[start : start + TARGET_SAMPLES]
            X.append(extract_features(chunk))
            y.append(LABEL_BACKGROUND)
            
            # Scaled noise chunk
            X.append(extract_features(chunk * np.random.uniform(0.4, 2.0)))
            y.append(LABEL_BACKGROUND)
            
    # Add synthetic noise transients (mic taps, clicks, hums, pink noise)
    for _ in range(80):
        noise_transient = generate_synthetic_noise_transient()
        X.append(extract_features(noise_transient))
        y.append(LABEL_BACKGROUND)
        
    # Add pure silence samples
    for _ in range(20):
        silence = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        X.append(extract_features(silence))
        y.append(LABEL_BACKGROUND)
            
    X = np.array(X, dtype=np.float32)
    X = X[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"Dataset generated: Total={len(X)} samples. Distribution: Keyword={np.sum(y == LABEL_KEYWORD)}, Unknown={np.sum(y == LABEL_UNKNOWN)}, Background={np.sum(y == LABEL_BACKGROUND)}")
    return X, y

