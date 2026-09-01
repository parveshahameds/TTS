import numpy as np

# IMA ADPCM constants
STEP_SIZE_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
]

INDEX_TABLE = [
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8
]

class ImaAdpcmEncoder:
    def __init__(self):
        self.predictor = 0
        self.step_index = 0
        
    def reset(self):
        self.predictor = 0
        self.step_index = 0
        
    def encode(self, pcm_samples):
        """
        Encodes 16-bit PCM samples to 4-bit IMA ADPCM bytes.
        Two 4-bit samples are packed into one byte.
        """
        # Ensure samples are int16 numpy array
        if pcm_samples.dtype != np.int16:
            # Scale to int16 range
            if np.max(np.abs(pcm_samples)) <= 1.0:
                pcm_samples = (pcm_samples * 32767).astype(np.int16)
            else:
                pcm_samples = pcm_samples.astype(np.int16)
                
        num_samples = len(pcm_samples)
        adpcm_len = (num_samples + 1) // 2
        adpcm_bytes = bytearray(adpcm_len)
        
        # State variables
        predictor = self.predictor
        step_index = self.step_index
        
        for i in range(0, num_samples, 2):
            # First sample (lower nibble)
            sample1 = pcm_samples[i]
            code1, predictor, step_index = self._encode_sample(sample1, predictor, step_index)
            
            # Second sample (upper nibble)
            if i + 1 < num_samples:
                sample2 = pcm_samples[i+1]
                code2, predictor, step_index = self._encode_sample(sample2, predictor, step_index)
            else:
                code2 = 0
                
            # Pack into one byte
            adpcm_bytes[i // 2] = (code2 << 4) | (code1 & 0x0F)
            
        self.predictor = predictor
        self.step_index = step_index
        return bytes(adpcm_bytes)
        
    def _encode_sample(self, sample, predictor, step_index):
        step = STEP_SIZE_TABLE[step_index]
        diff = sample - predictor
        
        # Determine sign bit
        if diff < 0:
            code = 8
            diff = -diff
        else:
            code = 0
            
        # Quantize difference
        temp_step = step
        if diff >= temp_step:
            code |= 4
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            code |= 2
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            code |= 1
            
        # Update predictor
        diff_pred = step >> 3
        if code & 4:
            diff_pred += step
        if code & 2:
            diff_pred += step >> 1
        if code & 1:
            diff_pred += step >> 2
            
        if code & 8:
            predictor = max(-32768, predictor - diff_pred)
        else:
            predictor = min(32767, predictor + diff_pred)
            
        # Update step index
        step_index += INDEX_TABLE[code & 0x0F]
        step_index = max(0, min(step_index, 88))
        
        return code, predictor, step_index

class ImaAdpcmDecoder:
    def __init__(self):
        self.predictor = 0
        self.step_index = 0
        
    def reset(self):
        self.predictor = 0
        self.step_index = 0
        
    def decode(self, adpcm_bytes, num_samples=None):
        """Decodes 4-bit IMA ADPCM bytes back to 16-bit PCM samples."""
        if num_samples is None:
            num_samples = len(adpcm_bytes) * 2
            
        pcm_samples = np.zeros(num_samples, dtype=np.int16)
        
        # State variables
        predictor = self.predictor
        step_index = self.step_index
        
        for i in range(0, num_samples, 2):
            byte_idx = i // 2
            if byte_idx >= len(adpcm_bytes):
                break
                
            val = adpcm_bytes[byte_idx]
            
            # First sample (lower nibble)
            code1 = val & 0x0F
            predictor, step_index = self._decode_sample(code1, predictor, step_index)
            pcm_samples[i] = predictor
            
            # Second sample (upper nibble)
            if i + 1 < num_samples:
                code2 = (val >> 4) & 0x0F
                predictor, step_index = self._decode_sample(code2, predictor, step_index)
                pcm_samples[i+1] = predictor
                
        self.predictor = predictor
        self.step_index = step_index
        return pcm_samples
        
    def _decode_sample(self, code, predictor, step_index):
        step = STEP_SIZE_TABLE[step_index]
        
        # Compute predicted difference
        diff_pred = step >> 3
        if code & 4:
            diff_pred += step
        if code & 2:
            diff_pred += step >> 1
        if code & 1:
            diff_pred += step >> 2
            
        # Update predictor
        if code & 8:
            predictor = max(-32768, predictor - diff_pred)
        else:
            predictor = min(32767, predictor + diff_pred)
            
        # Update step index
        step_index += INDEX_TABLE[code & 0x0F]
        step_index = max(0, min(step_index, 88))
        
        return predictor, step_index
