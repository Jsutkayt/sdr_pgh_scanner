# RadioTranscriber - Local File Version

This is a modified version of RadioTranscriber that processes pre-recorded audio files instead of streaming from an RTL-SDR device.

## Key Changes

### What's Different:
1. **Removed SDR streaming** - No longer uses `rtl_fm` or `sox` for live radio capture
2. **Added file processing** - Reads audio files from a `recordings/` folder
3. **Batch processing** - Processes all audio files in the folder sequentially
4. **Multiple format support** - Handles WAV, MP3, FLAC, OGG, M4A, AAC, WMA
5. **Automatic resampling** - Converts any sample rate to 16kHz for Whisper
6. **Stereo to mono** - Automatically converts stereo files to mono

### What's Kept:
- All transcription quality settings (Whisper model, beam search, etc.)
- Post-processing filters (beep detection, unit normalization, hallucination removal)
- Unit mapping and normalization
- Logging format and output structure

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Recordings Folder
```bash
mkdir recordings
```

### 3. Add Your Audio Files
Place your audio recordings in the `recordings/` folder. Supported formats:
- WAV
- MP3
- FLAC
- OGG
- M4A
- AAC
- WMA

### 4. Configure (Optional)
Edit `config.yaml` to adjust:
- Whisper model size (default: large-v3)
- Language (default: en)
- Post-processing filters
- Unit mappings for your area

## Usage

### Basic Usage
```bash
python transcribe_recordings.py
```

The script will:
1. Load all audio files from `recordings/`
2. Process them one by one
3. Save transcriptions to the output folder specified in `config.yaml`

### Output
Transcriptions are saved to a log file in the format:
```
[HH:MM:SS] (duration) [filename] Transcribed text here
```

Example:
```
[14:23:45] (12.3s) [dispatch_001.wav] Station 52, respond to 123 Main Street
[14:24:10] (8.5s) [dispatch_002.wav] 52A1 responding, in route
```

## Performance Tips

1. **Model Size**: 
   - `tiny` - Fastest, least accurate
   - `base` - Fast, decent for clear audio
   - `small` - Good balance
   - `medium` - Better quality, slower
   - `large-v3` - Best quality, slowest (recommended)

2. **GPU Usage**: 
   - CUDA-enabled GPU highly recommended for large models
   - CPU processing is very slow for large models

3. **Batch Size**: 
   - The script processes files sequentially
   - Garbage collection runs every 10 files

## Troubleshooting

### "No audio files found"
- Check that your files are in the `recordings/` folder
- Verify file extensions are supported
- Check file permissions

### "Error loading file"
- File may be corrupted
- Format may not be supported by soundfile
- Try converting to WAV with: `ffmpeg -i input.mp3 output.wav`

### Poor transcription quality
- Increase `beam_size` in config.yaml (default: 5, try 10)
- Increase `best_of` in config.yaml (default: 5, try 10)
- Use a larger model (large-v3 recommended)
- Adjust `initial_prompt` to include common terms from your area

### Out of memory
- Use a smaller model (medium or small)
- Process fewer files at once
- Close other applications

## Comparison to Original

| Feature | Original (SDR) | Modified (Files) |
|---------|---------------|------------------|
| Input | Live RTL-SDR stream | Pre-recorded audio files |
| Processing | Real-time | Batch |
| Dependencies | rtl_fm, sox, webrtcvad | soundfile |
| VAD | WebRTC VAD | Not needed |
| Format | Raw SDR IQ data | Common audio formats |
| Use Case | Live monitoring | Archive processing |

## Configuration Notes

Since you're not using SDR, these config.yaml sections are **ignored**:
- `sdr.frequency`
- `sdr.modulation`
- `sdr.sample_rate`
- `sdr.squelch`
- `sdr.gain`
- `sdr.ppm`
- `vad_and_silence.vad_aggressiveness`
- `vad_and_silence.silence_limit`

These are still **used**:
- `sdr.description` - Used for log naming
- `feed_specific.output_folder` - Where logs are saved
- `vad_and_silence.min_speech_seconds` - Minimum file duration to process
- All `tuning.*` settings
- All `post_generation_cleanup.*` settings

## Next Steps

1. Test with a small batch of files first
2. Review transcription quality
3. Adjust config.yaml settings as needed
4. Process your full archive

## Support

Based on: https://github.com/Nite01007/RadioTranscriber
Modified for local file processing