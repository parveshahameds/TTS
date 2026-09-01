import numpy as np
import speech_recognition as sr

class LocalASREngine:
    def __init__(self):
        """Initializes the Speech Recognition client."""
        self.recognizer = sr.Recognizer()
        
    def transcribe(self, pcm_data, sample_rate=16000):
        """
        Transcribes a 1D int16 PCM numpy array into a text string.
        
        Args:
            pcm_data (np.ndarray): 1D int16 PCM audio array.
            sample_rate (int): Sample rate of the audio (16000Hz).
        Returns:
            text (str): Transcription of the audio chunk.
        """
        # Ensure correct type
        if pcm_data.dtype != np.int16:
            if np.max(np.abs(pcm_data)) <= 1.0:
                pcm_data = (pcm_data * 32767).astype(np.int16)
            else:
                pcm_data = pcm_data.astype(np.int16)
                
        raw_bytes = pcm_data.tobytes()
        audio_data = sr.AudioData(raw_bytes, sample_rate, 2) # 2 bytes per sample (16-bit)
        
        # Try Google Speech API (free online, high accuracy)
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "" # Speech was detected but could not be parsed into words
        except sr.RequestError as re:
            print(f"ASR Cloud Request Failed: {re}. Attempting local PocketSphinx fallback...")
            # Fallback to local pocketsphinx (offline)
            try:
                text = self.recognizer.recognize_sphinx(audio_data)
                return text
            except Exception as se:
                return f"[Error: Internet offline & Sphinx failed: {se}]"
        except Exception as e:
            # Any other exception (like pocketsphinx not installed)
            try:
                # Try offline pocketsphinx directly
                return self.recognizer.recognize_sphinx(audio_data)
            except Exception:
                return "[Speech not understood / Offline Sphinx missing]"
                
if __name__ == "__main__":
    # Test with synthetic silence
    engine = LocalASREngine()
    silence = np.zeros(16000, dtype=np.int16)
    print("Testing transcription on silence:")
    print(f"Result: '{engine.transcribe(silence)}'")
