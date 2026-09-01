import os
import sys
import time
import numpy as np

# We'll use sounddevice and soundfile which are standard
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Error: sounddevice and soundfile must be installed to run this script.")
    print("Please make sure you run the script using the virtual environment python: ./.venv/bin/python")
    sys.exit(1)

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = 'PCM_16'

def record_audio(duration, filename):
    """Records audio from the default microphone for a given duration in seconds."""
    print(f"Recording for {duration} seconds...")
    try:
        # Record audio
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        
        # Save WAV file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        sf.write(filename, audio_data, SAMPLE_RATE, subtype=SUBTYPE)
        print(f"Saved: {filename}")
        return True
    except Exception as e:
        print(f"Error recording audio: {e}")
        return False

def interactive_record():
    print("====================================================")
    print("            EdgeWake Dataset Recorder               ")
    print("====================================================")
    print("This tool will guide you through recording audio samples.")
    print("Audio format: 16 kHz, Mono, 16-bit PCM WAV.")
    print("====================================================")

    data_dir = "data"
    os.makedirs(os.path.join(data_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "negative"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "background"), exist_ok=True)

    while True:
        print("\nSelect an option:")
        print("1. Record POSITIVE samples (Keyword: 'Hey Nova') [1.5 seconds each]")
        print("2. Record NEGATIVE samples (Acoustically similar or random words) [1.5 seconds each]")
        print("3. Record BACKGROUND noise (Silence, keyboard, fan, room noise) [5.0 seconds each]")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            count = input("How many positive samples would you like to record? (e.g. 15): ").strip()
            count = int(count) if count.isdigit() else 15
            print("\nInstructions: Speak the keyword 'Hey Nova' clearly. Try varying your tone, speed, and distance.")
            for i in range(count):
                timestamp = int(time.time() * 1000)
                filename = os.path.join(data_dir, "positive", f"positive_{timestamp}.wav")
                input(f"\n[Sample {i+1}/{count}] Press ENTER, wait for the prompt, then speak 'Hey Nova'...")
                record_audio(1.5, filename)
                time.sleep(0.5)
                
        elif choice == '2':
            # Suggested hard negatives
            negatives_list = [
                "Hello Nova", "Supernova", "Hey Nila", "Nova star", "Hey Bob", 
                "No way", "Yes please", "Open the door", "What is the time", "Nova"
            ]
            print("\nInstructions: You will record phrases that are similar to 'Hey Nova' or random speech.")
            print("Suggested phrases will be displayed.")
            count = len(negatives_list)
            for i, phrase in enumerate(negatives_list):
                timestamp = int(time.time() * 1000)
                filename = os.path.join(data_dir, "negative", f"negative_{timestamp}.wav")
                input(f"\n[Sample {i+1}/{count}] Phrase to speak: \"{phrase}\"\nPress ENTER, wait for the prompt, then speak...")
                record_audio(1.5, filename)
                time.sleep(0.5)
                
            # Ask if they want to record more custom negatives
            extra = input("\nDo you want to record custom negative speech? (y/n): ").strip().lower()
            if extra == 'y':
                extra_count = input("How many custom negative samples? ").strip()
                extra_count = int(extra_count) if extra_count.isdigit() else 5
                for i in range(extra_count):
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(data_dir, "negative", f"negative_{timestamp}.wav")
                    input(f"\n[Custom Negative {i+1}/{extra_count}] Press ENTER, wait for the prompt, then speak random speech...")
                    record_audio(1.5, filename)
                    time.sleep(0.5)

        elif choice == '3':
            backgrounds = [
                ("silence", "Complete silence / ambient room noise"),
                ("keyboard", "Keyboard typing / clicking sounds"),
                ("environment", "Background environment noise (fan, music, street, talk)")
            ]
            print("\nInstructions: Record longer ambient/noise clips. Do NOT speak during these.")
            for name, desc in backgrounds:
                count = input(f"\nHow many {name} clips ({desc})? (e.g. 3): ").strip()
                count = int(count) if count.isdigit() else 3
                for i in range(count):
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(data_dir, "background", f"background_{name}_{timestamp}.wav")
                    input(f"\n[{name} {i+1}/{count}] Setup: {desc}.\nPress ENTER when ready to record for 5 seconds...")
                    record_audio(5.0, filename)
                    time.sleep(0.5)

        elif choice == '4':
            print("Exiting Dataset Recorder.")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    interactive_record()
