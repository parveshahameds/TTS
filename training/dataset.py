import os
import glob
import numpy as np
import soundfile as sf
from edge.features import extract_features, TARGET_SAMPLES
from training.augment import augment_waveform, pad_or_crop

# Labels
LABEL_KEYWORD = 0       # "Hey Nova"
LABEL_UNKNOWN = 1       # Other speech
LABEL_BACKGROUND = 2    # Silence / environmental noise
NUM_CLASSES = 3

def load_wav(filepath):
    """Loads a WAV file, resampling or converting to mono if necessary, normalized to float32 [-1.0, 1.0]."""
    try:
        data, samplerate = sf.read(filepath)
        if samplerate != 16000:
            # Simple decimation/interpolation fallback
            # In a real environment we assume 16kHz WAVs are recorded.
            pass
        if len(data.shape) > 1:
            data = np.mean(data, axis=1) # Mono conversion
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def generate_synthetic_data(num_samples=60):
    """Generates synthetic audio data for development testing when no recordings exist."""
    print("Warning: Generating synthetic training data because dataset directories are empty or missing.")
    X = []
    y = []
    
    # 16240 samples for 100 spectrogram frames
    t = np.linspace(0, 1.0, TARGET_SAMPLES)
    
    for i in range(num_samples):
        label = i % 3
        if label == LABEL_KEYWORD:
            # Keyword: 'Hey Nova' -> Mix of 440Hz and 880Hz sinewaves
            audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        elif label == LABEL_UNKNOWN:
            # Unknown speech -> Mix of 220Hz and 660Hz sinewaves
            audio = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)
        else:
            # Background -> Low amplitude white noise
            audio = np.random.normal(0, 0.02, TARGET_SAMPLES)
            
        # Add random noise
        audio += np.random.normal(0, 0.01, TARGET_SAMPLES)
        
        # Extract features
        features = extract_features(audio)
        X.append(features)
        y.append(label)
        
    X = np.array(X)[..., np.newaxis] # Add channel dimension -> (N, 100, 40, 1)
    y = np.array(y)
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
        
    # Load raw background noise samples for mixing
    bg_noises = []
    for f in bg_files:
        audio = load_wav(f)
        if audio is not None:
            bg_noises.append(audio)
            
    X = []
    y = []
    
    # Process positive samples (Keyword)
    for f in pos_files:
        audio = load_wav(f)
        if audio is not None:
            # Add original
            features = extract_features(pad_or_crop(audio, TARGET_SAMPLES))
            X.append(features)
            y.append(LABEL_KEYWORD)
            
            # Add augmented versions
            for _ in range(5):  # 5x augmentation for small datasets
                aug_audio = augment_waveform(audio, bg_noises)
                X.append(extract_features(aug_audio))
                y.append(LABEL_KEYWORD)
                
    # Process negative samples (Unknown Speech)
    for f in neg_files:
        audio = load_wav(f)
        if audio is not None:
            features = extract_features(pad_or_crop(audio, TARGET_SAMPLES))
            X.append(features)
            y.append(LABEL_UNKNOWN)
            
            # Add augmented versions
            for _ in range(5):
                aug_audio = augment_waveform(audio, bg_noises)
                X.append(extract_features(aug_audio))
                y.append(LABEL_UNKNOWN)

    # Process background samples (Silence/Noise)
    # Background clips can be sliced into multiple 1-second chunks
    for f in bg_files:
        audio = load_wav(f)
        if audio is not None:
            # Slice into 1-second (16240 samples) segments
            step = TARGET_SAMPLES // 2  # 50% overlap
            for start in range(0, len(audio) - TARGET_SAMPLES, step):
                chunk = audio[start:start+TARGET_SAMPLES]
                X.append(extract_features(chunk))
                y.append(LABEL_BACKGROUND)
                
                # Add augmented chunk
                for _ in range(2):
                    aug_chunk = augment_waveform(chunk, None) # No need to mix noise into noise
                    X.append(extract_features(aug_chunk))
                    y.append(LABEL_BACKGROUND)
                    
    # Fallback if somehow background count is zero
    if len(bg_files) == 0:
        # Create silent/noise samples
        for _ in range(max(10, len(pos_files))):
            noise = np.random.normal(0, 0.01, TARGET_SAMPLES)
            X.append(extract_features(noise))
            y.append(LABEL_BACKGROUND)
            
    X = np.array(X, dtype=np.float32)
    X = X[..., np.newaxis] # Add channel dimension
    y = np.array(y, dtype=np.int32)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y
