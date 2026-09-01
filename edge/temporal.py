import time

class TemporalVerifier:
    def __init__(self, threshold=0.7, min_consecutive_frames=3, smoothing_factor=0.7, cooldown_seconds=2.0):
        """
        Temporal Verification for confidence smoothing.
        
        Args:
            threshold (float): Confidence threshold (0.0 to 1.0) to consider a frame active.
            min_consecutive_frames (int): Minimum frames above threshold required to trigger.
            smoothing_factor (float): Exponential smoothing factor (alpha) where:
                                     S_t = alpha * S_{t-1} + (1 - alpha) * X_t
            cooldown_seconds (float): Cooldown time in seconds after a trigger to prevent multi-triggering.
        """
        self.threshold = threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.smoothing_factor = smoothing_factor
        self.cooldown_seconds = cooldown_seconds
        
        self.smoothed_prob = 0.0
        self.consecutive_count = 0
        self.last_trigger_time = 0.0
        
    def process_probability(self, raw_prob):
        """
        Processes a new keyword probability estimate.
        Returns:
            is_triggered (bool): True if KWS is temporally verified.
            smoothed_prob (float): The smoothed probability score.
        """
        # 1. Apply Exponential Smoothing
        self.smoothed_prob = (self.smoothing_factor * self.smoothed_prob) + ((1.0 - self.smoothing_factor) * raw_prob)
        
        # 2. Check threshold
        if self.smoothed_prob > self.threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1) # Slowly decay count instead of hard reset to zero
            
        # 3. Check trigger condition
        is_triggered = False
        current_time = time.time()
        
        if self.consecutive_count >= self.min_consecutive_frames:
            # Check if cooldown has elapsed
            if current_time - self.last_trigger_time > self.cooldown_seconds:
                is_triggered = True
                self.last_trigger_time = current_time
            # Reset consecutive count after a trigger
            self.consecutive_count = 0
            self.smoothed_prob = 0.0
            
        return is_triggered, self.smoothed_prob
        
    def reset(self):
        """Resets the internal smoothing states."""
        self.smoothed_prob = 0.0
        self.consecutive_count = 0
