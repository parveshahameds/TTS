import numpy as np

class AudioRingBuffer:
    def __init__(self, capacity=8000):
        """
        Circular ring buffer for pre-roll audio.
        
        Args:
            capacity (int): Number of audio samples to store (8000 samples = 500ms at 16kHz).
        """
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.int16)
        self.write_index = 0
        self.is_full = False
        
    def write(self, data):
        """Writes a chunk of numpy audio data into the circular buffer."""
        length = len(data)
        if length >= self.capacity:
            self.buffer = data[-self.capacity:].copy()
            self.write_index = 0
            self.is_full = True
            return
            
        space_left = self.capacity - self.write_index
        if length <= space_left:
            self.buffer[self.write_index:self.write_index+length] = data
            self.write_index += length
        else:
            self.buffer[self.write_index:] = data[:space_left]
            self.buffer[:length - space_left] = data[space_left:]
            self.write_index = length - space_left
            self.is_full = True
            
    def get_content(self):
        """Retrieves the accumulated audio in chronological order."""
        if not self.is_full:
            return self.buffer[:self.write_index].copy()
        # Concatenate from write_index to end, then from beginning to write_index
        return np.concatenate((self.buffer[self.write_index:], self.buffer[:self.write_index]))
        
    def clear(self):
        """Clears the buffer."""
        self.buffer.fill(0)
        self.write_index = 0
        self.is_full = False
