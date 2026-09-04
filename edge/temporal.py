import time

class TemporalVerifier:
    def __init__(self, threshold=0.80, min_consecutive_frames=3, smoothing_factor=0.5, margin=0.20, cooldown_seconds=2.0):
        """
        Temporal Verification for confidence smoothing and false trigger elimination.
        
        Args:
            threshold (float): Confidence threshold (0.0 to 1.0) required for keyword activation.
            min_consecutive_frames (int): Consecutive verified frames required to trigger.
            smoothing_factor (float): Exponential smoothing factor (alpha).
            margin (float): Required minimum margin that keyword prob must exceed non-keyword probs.
            cooldown_seconds (float): Cooldown time in seconds after a trigger.
        """
        self.threshold = threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.smoothing_factor = smoothing_factor
        self.margin = margin
        self.cooldown_seconds = cooldown_seconds
        
        self.smoothed_prob = 0.0
        self.consecutive_count = 0
        self.last_trigger_time = 0.0
        
    def process_probability(self, probs):
        """
        Processes 3-class probability distribution [P(Keyword), P(Unknown), P(Background)]
        or a single scalar float P(Keyword).
        
        Returns:
            is_triggered (bool): True if KWS is temporally verified.
            smoothed_prob (float): The smoothed keyword probability score.
        """
        if isinstance(probs, (list, tuple)) or (hasattr(probs, 'shape') and len(probs) >= 3):
            kw_prob = float(probs[0])
            unk_prob = float(probs[1])
            bg_prob = float(probs[2])
            
            # Check margin: Keyword must clearly beat Unknown speech and Background noise
            is_valid_frame = (
                kw_prob >= self.threshold and 
                kw_prob > (unk_prob + self.margin) and 
                kw_prob > (bg_prob + self.margin)
            )
        else:
            kw_prob = float(probs)
            is_valid_frame = kw_prob >= self.threshold
            
        # 1. Apply Exponential Smoothing
        self.smoothed_prob = (self.smoothing_factor * self.smoothed_prob) + ((1.0 - self.smoothing_factor) * kw_prob)
        
        # 2. Update consecutive match counter with hard reset on invalid frames
        if is_valid_frame and self.smoothed_prob >= (self.threshold * 0.9):
            self.consecutive_count += 1
        else:
            # Immediate hard reset to avoid leaky noise accumulation over time
            self.consecutive_count = 0
            
        # 3. Check trigger condition
        is_triggered = False
        current_time = time.time()
        
        if self.consecutive_count >= self.min_consecutive_frames:
            if current_time - self.last_trigger_time > self.cooldown_seconds:
                is_triggered = True
                self.last_trigger_time = current_time
            self.consecutive_count = 0
            self.smoothed_prob = 0.0
            
        return is_triggered, self.smoothed_prob
        
    def reset(self):
        """Resets the internal smoothing states."""
        self.smoothed_prob = 0.0
        self.consecutive_count = 0

