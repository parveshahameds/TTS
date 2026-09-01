import queue
import numpy as np
import sounddevice as sd

class AudioMicrophoneStream:
    def __init__(self, sample_rate=16000, block_size=160):
        """
        Microphone Stream using sounddevice.
        
        Args:
            sample_rate (int): Sample rate (16000Hz).
            block_size (int): Size of audio block returned (160 samples = 10ms frame).
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.q = queue.Queue()
        self.stream = None
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback function called by sounddevice for each new block of audio."""
        if status:
            print(f"Audio Stream status: {status}")
        # Capture mono audio as int16
        # sounddevice returns float32 by default unless specified
        # Let's request int16 directly
        self.q.put(indata[:, 0].copy())
        
    def start(self):
        """Starts the microphone audio stream."""
        if self.stream is not None:
            return
            
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.block_size,
            callback=self._audio_callback
        )
        self.stream.start()
        print("Microphone stream started.")
        
    def stop(self):
        """Stops the audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Microphone stream stopped.")
            
    def read(self, block=True, timeout=None):
        """Reads a chunk of audio from the queue."""
        try:
            return self.q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def clear(self):
        """Clears any remaining frames in the queue."""
        with self.q.mutex:
            self.q.queue.clear()
            
if __name__ == "__main__":
    import time
    stream = AudioMicrophoneStream()
    stream.start()
    try:
        print("Recording for 3 seconds...")
        time.sleep(3.0)
        print(f"Recorded {stream.q.qsize()} frames of size {stream.block_size}.")
    finally:
        stream.stop()
