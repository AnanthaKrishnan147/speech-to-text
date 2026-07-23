import os
import re
import wave
import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer

# 1. SETUP CONFIG
OUTPUT_WAV_FILENAME = "mms_debug.wav"
TEXT_TO_TEST = "कृത്രിമബുദ്ധി മനുഷ്യന്റെ ജീവിതത്തെ ലളിതമാക്കുന്നു! ഇതാണ് പുതിയ സാങ്കേതികവിദ്യ."
LANG_TAG = "ML"

LANGUAGE_MMS_REPOS = {
    "ML": "facebook/mms-tts-mal",
}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sample_rate = 16000  # Meta MMS outputs exactly at 16000Hz

# 2. LOAD MODEL
repo_id = LANGUAGE_MMS_REPOS[LANG_TAG]
print(f"[debug] Loading Meta MMS model ({repo_id}) on {device}...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = VitsModel.from_pretrained(repo_id).to(device)
model.eval()

# 3. PREPARE WAV FILE STRUCT
wav_file = wave.open(OUTPUT_WAV_FILENAME, "wb")
wav_file.setnchannels(1)      # Mono audio
wav_file.setsampwidth(2)      # 16-bit depth (2 bytes per sample)
wav_file.setframerate(sample_rate)

# 4. SIMULATE PIPELINE SENTENCE SPLITTING
_SPLIT_RE = re.compile(r'([.!?\n])')
parts = _SPLIT_RE.split(TEXT_TO_TEST)
sentences = []

# Reconstruct strings matched by split groups
for i in range(0, len(parts) - 1, 2):
    phrase = parts[i] + parts[i + 1]
    if phrase.strip():
        sentences.append(phrase.strip())
if parts[-1].strip():
    sentences.append(parts[-1].strip())

print(f"[debug] Parsed text into {len(sentences)} sentence chunk(s) to process.")

# 5. GENERATE AND CONCATENATE INTO WAV
for idx, sentence in enumerate(sentences, start=1):
    if len(sentence) < 4:  # Clean threshold check
        continue
        
    print(f" -> Processing sentence chunk {idx}/{len(sentences)}: {sentence!r}")
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        audio_data = outputs.waveform[0].cpu().numpy()
        
    # Normalize amplitude
    peak = np.max(np.abs(audio_data))
    if peak == 0: peak = 1.0
    audio_chunk = (audio_data / peak) * 0.9

    # Convert float32 down to 16-bit signed int PCM payload
    audio_int16 = (audio_chunk * 32768).astype(np.int16)
    
    # Append frames seamlessly behind the preceding sentence
    wav_file.writeframes(audio_int16.tobytes())

wav_file.close()
print(f"\n✅ Finished! Sentence vectors combined and exported to: {os.path.abspath(OUTPUT_WAV_FILENAME)}")