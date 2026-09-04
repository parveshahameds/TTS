import numpy as np

SAMPLE_RATE = 16000
TARGET_SAMPLES = 16240

def generate_synthetic_phoneme_speech():
    """
    Generates a realistic multi-syllable synthetic speech utterance with 
    formant transitions, harmonic pulses, pitch modulation, and syllable envelopes.
    """
    duration = np.random.uniform(0.5, 0.95)
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Fundamental pitch with subtle vibrato/contour
    base_f0 = np.random.uniform(90, 260)
    pitch_contour = base_f0 + 15 * np.sin(2 * np.pi * np.random.uniform(1, 4) * t)
    
    num_syllables = np.random.randint(2, 5)
    syllable_len = num_samples // num_syllables
    
    utterance = np.zeros(num_samples, dtype=np.float32)
    
    for s in range(num_syllables):
        s_start = s * syllable_len
        s_end = min(num_samples, s_start + syllable_len)
        s_t = t[s_start:s_end]
        if len(s_t) == 0:
            continue
            
        # Formant frequencies for diverse vowels
        f1 = np.random.uniform(280, 850)
        f2 = np.random.uniform(850, 2300)
        f3 = np.random.uniform(2100, 3200)
        
        # Glottal excitation
        phase = 2 * np.pi * np.cumsum(pitch_contour[s_start:s_end]) / SAMPLE_RATE
        glottal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
        
        # Formant resonances
        h1 = np.sin(2 * np.pi * f1 * s_t) * 0.5
        h2 = np.sin(2 * np.pi * f2 * s_t) * 0.3
        h3 = np.sin(2 * np.pi * f3 * s_t) * 0.2
        
        vowel = glottal * (h1 + h2 + h3)
        
        # Consonant burst (fricative or plosive) at syllable onset
        if np.random.random() < 0.6:
            burst_len = min(len(s_t), int(np.random.uniform(0.02, 0.06) * SAMPLE_RATE))
            burst = np.random.normal(0, 0.3, burst_len) * np.linspace(1, 0, burst_len)
            vowel[:burst_len] += burst
            
        # Syllable attack-decay envelope
        env = np.sin(np.pi * np.linspace(0, 1, len(s_t))) ** 2
        utterance[s_start:s_end] = vowel * env * np.random.uniform(0.3, 0.7)
        
    # Place within the 16240 target window
    padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    start_pos = np.random.randint(0, max(1, TARGET_SAMPLES - num_samples + 1))
    padded[start_pos:start_pos + num_samples] = utterance
    return padded

def generate_synthetic_confuser():
    """
    Generates speech-like tokens that have partial phonetic similarities to 'Hey Nova'
    (e.g., 'Hey', 'Nova', 'No', 'Never', 'Dave', 'Keva', 'Over', 'Hello') but are NOT the keyword.
    """
    patterns = ['hey_alone', 'nova_alone', 'hello', 'never', 'over', 'dave', 'radio']
    pattern = np.random.choice(patterns)
    
    if pattern == 'hey_alone':
        # Single syllable ~300ms
        dur = 0.35
        num_s = int(dur * SAMPLE_RATE)
        t = np.linspace(0, dur, num_s, endpoint=False)
        f0 = np.random.uniform(110, 200)
        phase = 2 * np.pi * f0 * t
        glottal = np.sin(phase) + 0.4 * np.sin(2 * phase)
        # /e/ -> /i/ transition
        f1 = np.linspace(550, 300, num_s)
        f2 = np.linspace(1800, 2300, num_s)
        vowel = glottal * (np.sin(2*np.pi*f1*t)*0.5 + np.sin(2*np.pi*f2*t)*0.3)
        # /h/ aspiration at start
        asp_len = int(0.08 * SAMPLE_RATE)
        vowel[:asp_len] += np.random.normal(0, 0.2, asp_len)
        env = np.sin(np.pi * np.linspace(0, 1, num_s))**1.5
        sig = vowel * env * 0.5
    elif pattern == 'nova_alone':
        # Two syllables: /no/ + /va/
        dur = 0.55
        num_s = int(dur * SAMPLE_RATE)
        t = np.linspace(0, dur, num_s, endpoint=False)
        f0 = np.random.uniform(120, 210)
        phase = 2 * np.pi * f0 * t
        glottal = np.sin(phase) + 0.4 * np.sin(2 * phase)
        # /o/ -> /a/
        f1 = np.concatenate([np.linspace(500, 550, num_s//2), np.linspace(700, 750, num_s - num_s//2)])
        f2 = np.concatenate([np.linspace(900, 950, num_s//2), np.linspace(1200, 1100, num_s - num_s//2)])
        vowel = glottal * (np.sin(2*np.pi*f1*t)*0.5 + np.sin(2*np.pi*f2*t)*0.3)
        env = np.concatenate([np.sin(np.pi * np.linspace(0, 1, num_s//2))**2, np.sin(np.pi * np.linspace(0, 1, num_s - num_s//2))**2])
        sig = vowel * env * 0.5
    else:
        return generate_synthetic_phoneme_speech()
        
    padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    start_pos = np.random.randint(0, max(1, TARGET_SAMPLES - len(sig) + 1))
    padded[start_pos:start_pos + len(sig)] = sig
    return padded

def generate_synthetic_noise_transient():
    """Generates acoustic noise transients: mic thumps, clicks, white noise bursts, hums."""
    noise_type = np.random.choice(['click', 'thump', 'burst', 'hum', 'pink'])
    sig = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    
    if noise_type == 'click':
        click_len = np.random.randint(50, 300)
        pos = np.random.randint(0, TARGET_SAMPLES - click_len)
        sig[pos:pos+click_len] = np.random.normal(0, np.random.uniform(0.3, 0.8), click_len)
    elif noise_type == 'thump':
        t_len = int(0.12 * SAMPLE_RATE)
        pos = np.random.randint(0, TARGET_SAMPLES - t_len)
        t = np.linspace(0, 0.12, t_len)
        thump = np.sin(2 * np.pi * np.random.uniform(40, 120) * t) * np.exp(-t * 25)
        sig[pos:pos+t_len] = thump * np.random.uniform(0.3, 0.7)
    elif noise_type == 'burst':
        b_len = int(0.20 * SAMPLE_RATE)
        pos = np.random.randint(0, TARGET_SAMPLES - b_len)
        burst = np.random.normal(0, np.random.uniform(0.05, 0.2), b_len)
        sig[pos:pos+b_len] = burst
    elif noise_type == 'hum':
        t = np.linspace(0, 1.015, TARGET_SAMPLES)
        f = np.random.choice([50.0, 60.0, 100.0, 120.0])
        sig = (0.5 * np.sin(2*np.pi*f*t) + 0.25 * np.sin(2*np.pi*2*f*t)) * np.random.uniform(0.01, 0.05)
    else:
        sig = np.random.normal(0, np.random.uniform(0.005, 0.03), TARGET_SAMPLES).astype(np.float32)
        
    return sig

