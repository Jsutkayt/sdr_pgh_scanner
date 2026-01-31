
import numpy as np
import subprocess
import tempfile
import datetime
import time
import queue
import threading
import scipy.signal as signal
import os
import gc
import re
from collections import Counter
import yaml
import soundfile as sf
from pathlib import Path

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Extract config values
FEED_DESCRIPTION = config["feed_specific"]["description"]
OUTPUT_FOLDER = config["feed_specific"]["output_folder"]

MIN_SPEECH_SECONDS = config["vad_and_silence"]["min_speech_seconds"]

MODEL_SIZE = config["tuning"]["model_size"]
LANGUAGE = config["tuning"]["language"]
INITIAL_PROMPT = config["tuning"]["initial_prompt"]
BEAM_SIZE = config["tuning"]["beam_size"]
BEST_OF = config["tuning"]["best_of"]
NO_SPEECH_THRESHOLD = config["tuning"]["no_speech_threshold"]
NORMALIZATION_PERCENTILE = config["tuning"]["normalization"]

FULL_BLOCK_PHRASES = config["post_generation_cleanup"]["full_block_phrases"]
CUTOFF_PHRASES = config["post_generation_cleanup"]["cutoff_phrases"]
UNIT_MAPPING = config["post_generation_cleanup"]["unit_mapping"]
UNIT_PATTERN = config["post_generation_cleanup"]["unit_normalization"]["pattern"]
UNIT_PREFIX = config["post_generation_cleanup"]["unit_normalization"]["prefix"]


CLI_PATH = os.path.expanduser(config["whisper_backend"]["cli_path"])
MODEL_PATH = os.path.expanduser(config["whisper_backend"]["model_path"])

# --- SETTINGS ---
SAMPLE_RATE = 16000
RECORDINGS_FOLDER = "recordings"  # Folder containing audio files

BASE_LOG_FILENAME = f"transcription_{FEED_DESCRIPTION}"
LOG_FILE = os.path.join(OUTPUT_FOLDER, f"{BASE_LOG_FILENAME}_{datetime.date.today()}.log")
CURRENT_LOG_DATE = datetime.date.today()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'a').close()

sos = signal.butter(5, 100 / (SAMPLE_RATE / 2), btype='high', output='sos')

transcription_queue = queue.Queue()


# --- INITIALIZATION --- Medium Model Works Best
#print(f"Loading Whisper model '{MODEL_SIZE}'...")
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model = whisper.load_model(MODEL_SIZE).to(device)
#print(f"Model loaded on {device}")

print(f"Processing recordings from '{RECORDINGS_FOLDER}' folder")
print(f"   Feed: {FEED_DESCRIPTION}")
print(f"   Output: {OUTPUT_FOLDER}")

def load_audio_file(filepath):
    """
    Load an audio file and convert to 16kHz mono float32
    Supports: WAV, MP3, FLAC, OGG, M4A
    """
    try:
        # Read audio file
        audio, sr = sf.read(filepath)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            from scipy import signal as sp_signal
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = sp_signal.resample(audio, num_samples)
        
        # Ensure float32 and normalize
        audio = audio.astype(np.float32)
        
        # Apply high-pass filter
        audio, _ = signal.sosfilt(sos, audio, zi=np.zeros((sos.shape[0], 2)))
        
        return audio
        
    except Exception as e:
        print(f"   Error loading {filepath}: {e}")
        return None

def transcriber_worker():
    print(f"   [Worker] Transcriber thread started")
    
    while True:
        try:
            item = transcription_queue.get()
            if item is None: 
                break
            
            timestamp, audio_data, filename = item

            duration = len(audio_data) / SAMPLE_RATE
            print(f"Transcribing {filename} ({duration:.1f}s)...")
            transcribe_start = time.time()
            
            #result = model.transcribe(
               # audio_data,
               # language=LANGUAGE,
               # fp16=(device == "cuda"),
               # initial_prompt=INITIAL_PROMPT,
               # condition_on_previous_text=False,
               # temperature=0.0,
               # beam_size=BEAM_SIZE,
               # best_of=BEST_OF,
              #  patience=1.5,
             #   suppress_blank=True
            #)
            
            #text = result['text'].strip()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete = False) as tmp_file:
                    temp_wav_path = tmp_file.name
                    sf.write(temp_wav_path, audio_data, SAMPLE_RATE)
            
            try:

                cmd = [
                    CLI_PATH,
                    "-m", MODEL_PATH,  #large-v3
                    "-f", temp_wav_path,
                    "-l", LANGUAGE,
                    "-bs", str(BEAM_SIZE),              # beam_size from config
                    "-bo", str(BEST_OF),              # best_of from config  
                    "-t", "4",               # threads
                    "--prompt", INITIAL_PROMPT,  # Your detailed prompt
                    "--temperature", "0.0",      # Deterministic output
                    "-nt"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

               # Extract text from whisper.cpp output
                output_lines = result.stdout.split('\n')
                text = ""

                # The transcription is after all the loading info
                # Look for lines that don't start with common prefixes
                for line in output_lines:
                    line = line.strip()
                    # Skip technical/metadata lines
                    if not line:
                        continue
                    if line.startswith(('whisper_', 'ggml_', 'system_info:', 'main:', '[')):
                        continue
                    # This should be the transcription
                    if line:
                        text = line
                        break
    
            finally:
    # Clean up temp file
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)


            transcribe_time = time.time() - transcribe_start
            print(f"   Done in {transcribe_time:.1f}s")

            original_text = text

            # Beep hallucination replacement
            if duration < 10.0:
                beep_patterns = [
                    "BEEE", "BEEEE", "EEEE", "BEEP", "BEEEP", 
                    "BEEEEEEEE", "EEEEEEEE", 
                    "AAAA", "AAAAA", "AAAAAA", "A A A", 
                    "AAAAAAAAA", "AAAAAAAAAA"
                ]
                upper_text = text.upper()
                if any(pattern in upper_text for pattern in beep_patterns) and len(text) > 10:
                    text = "[beeps]"
                    print(f"   (Replaced alert tone: {original_text} → [beeps])")

            # Long single-char run → [noise]
            if re.search(r'([A-Z])\1{10,}', text.upper()):
                text = "[noise]"
                print(f"   (Replaced noise run: {original_text} → [noise])")

            # De-duplicate repeated unit calls
            words = text.split()
            if len(words) > 3:
                common = Counter(words).most_common(1)
                if common and len(common[0][0]) <= 10 and common[0][1] > len(words) // 2:
                    text = common[0][0]
                    print(f"   (De-duplicated repetition: {original_text} → {text})")

            # Map spoken numbers to letter units
            lower_text = text.lower()
            for spoken, letter in UNIT_MAPPING.items():
                if spoken in lower_text:
                    text = re.sub(re.escape(spoken), letter, text, flags=re.IGNORECASE)
                    print(f"   (Mapped unit: {original_text} → {text})")

            # Normalize hyphens in unit IDs
            def normalize_unit(match):
                letter = match.group(1).upper()
                num = match.group(2)
                return f"{UNIT_PREFIX}{letter}{num}"
            
            text = re.sub(UNIT_PATTERN, normalize_unit, text, flags=re.IGNORECASE)
            if text != original_text:
                print(f"   (Normalized units: {original_text} → {text})")

            lower_text = text.lower()
            blocked = False
            for phrase in FULL_BLOCK_PHRASES:
                if phrase.lower() in lower_text:
                    print(f"   (Blocked hallucination containing '{phrase}': {text})")
                    blocked = True
                    break
            if blocked:
                transcription_queue.task_done()
                continue

            for phrase in CUTOFF_PHRASES:
                idx = lower_text.find(phrase.lower())
                if idx != -1:
                    text = text[:idx].strip()
                    print(f"   (Truncated hallucinated phrase '{phrase}': {original_text})")
                    break

            # Capitalization for clarity
            keywords = r'(dispatch|central|station 52|baystate|wing|cooley|amr|bravo \d+|engine \d+|ladder \d+|a\d+|e\d+|b\d+|bs\d+|bl\d+)'
            text = re.sub(keywords, lambda m: m.group(0).title(), text, flags=re.I)

            if text:
                output = f"[{timestamp}] ({duration:.1f}s) [{filename}] {text}"
                print(output)
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(output + "\n")
            else:
                print(f"   (Empty after cleanup — discarded)")

            transcription_queue.task_done()

        except Exception as e:
            print(f"Error in transcriber: {e}")

def process_recordings():
    global LOG_FILE, CURRENT_LOG_DATE
    
    # Start transcriber worker thread
    worker = threading.Thread(target=transcriber_worker)
    worker.daemon = True
    worker.start()
    
    # Get list of audio files
    recordings_path = Path(RECORDINGS_FOLDER)
    if not recordings_path.exists():
        print(f"Error: '{RECORDINGS_FOLDER}' folder not found!")
        print(f"Please create the folder and add your audio files.")
        return
    
    # Supported audio formats
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(recordings_path.glob(f'*{ext}'))
        audio_files.extend(recordings_path.glob(f'*{ext.upper()}'))
    
    audio_files = sorted(set(audio_files))  # Remove duplicates and sort
    
    if not audio_files:
        print(f"No audio files found in '{RECORDINGS_FOLDER}'")
        print(f"Supported formats: {', '.join(audio_extensions)}")
        return
    
    print(f"\nFound {len(audio_files)} audio file(s) to process")
    print("=" * 60)
    
    start_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{start_timestamp}] [STARTED] Batch transcription - {len(audio_files)} files\n")
    
    try:
        for idx, filepath in enumerate(audio_files, 1):
            print(f"\n[{idx}/{len(audio_files)}] Loading: {filepath.name}")
            
            # Load audio file
            audio_data = load_audio_file(str(filepath))
            
            if audio_data is None:
                continue
            
            duration = len(audio_data) / SAMPLE_RATE
            
            # Skip files that are too short
            if duration < MIN_SPEECH_SECONDS:
                print(f"   (Skipping: too short {duration:.1f}s)")
                continue
            
            # Normalize audio
            if len(audio_data) > 0:
                percentile_val = np.percentile(np.abs(audio_data), NORMALIZATION_PERCENTILE)
                if percentile_val > 0:
                    audio_data = audio_data / percentile_val
                    audio_data = np.clip(audio_data, -1.0, 1.0)
            
            audio_data = audio_data.astype(np.float32)
            
            # Queue for transcription
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            transcription_queue.put((timestamp, audio_data.copy(), filepath.name))
            print(f"   Queued for transcription ({duration:.1f}s)")
            
            # Periodic garbage collection
            if idx % 10 == 0:
                gc.collect()
        
        # Wait for all transcriptions to complete
        print("\n" + "=" * 60)
        print("Waiting for transcriptions to complete...")
        transcription_queue.join()
        
    except KeyboardInterrupt:
        print("\n\nStopping transcriptor...")
    finally:
        # Stop worker thread
        transcription_queue.put(None)
        worker.join()
        
        stop_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{stop_timestamp}] [COMPLETED] Batch transcription finished")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{stop_timestamp}] [COMPLETED] Batch transcription finished\n")
        
        print(f"\nTranscriptions saved to: {LOG_FILE}")

if __name__ == "__main__":
    process_recordings()