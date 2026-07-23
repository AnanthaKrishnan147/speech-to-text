import os
import re
import sys
import queue
import threading
import time
import wave
import numpy as np
import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer

# 1. SETUP CONFIG
OUTPUT_WAV_FILENAME = "parler_debug.wav"
PLAY_STEPS = 30
TEXT_TO_TEST = "കൃത്രിമബുദ്ധി മനുഷ്യന്റെ ജീവിതത്തെ ലളിതമാക്കുന്നു. ഇതാണ് പുതിയ സാങ്കേതികവിദ്യ."
LANG_TAG = "ML"

LANGUAGE_TTS_CONFIG = {
    "ML": {"speaker_desc": "Anjali speaks in a Conversation style with a warm, natural, and expressive tone, at a normal, relaxed pace as if speaking casually to a friend."}
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

# 2. LOAD MODELS
print(f"[debug] Loading Parler-TTS on {device}...")
repo_id = "ai4bharat/indic-parler-tts"
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(repo_id)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
sample_rate = model.config.sampling_rate

# 3. PREPARE WAV FILE STRUCT
# Open the file in write-binary mode
wav_file = wave.open(OUTPUT_WAV_FILENAME, "wb")
wav_file.setnchannels(1)      # Mono audio
wav_file.setsampwidth(2)      # 16-bit depth (2 bytes per sample)
wav_file.setframerate(sample_rate)

# 4. STREAM GENERATION AND SAVE
print(f"[debug] Starting streaming inference for text: {TEXT_TO_TEST!r}")
description_inputs = description_tokenizer(LANGUAGE_TTS_CONFIG[LANG_TAG]["speaker_desc"], return_tensors="pt").to(device)
prompt_inputs = tokenizer(TEXT_TO_TEST, return_tensors="pt").to(device)

streamer = ParlerTTSStreamer(model, device=device, play_steps=PLAY_STEPS)

generation_kwargs = dict(
    input_ids=description_inputs.input_ids,
    attention_mask=description_inputs.attention_mask,
    prompt_input_ids=prompt_inputs.input_ids,
    prompt_attention_mask=prompt_inputs.attention_mask,
    streamer=streamer,
    do_sample=True,
    temperature=0.9,
)

def _run_generate():
    try:
        model.generate(**generation_kwargs)
    except Exception as e:
        print(f"[debug] Generation thread error: {e}")

gen_thread = threading.Thread(target=_run_generate)
t0 = time.time()
gen_thread.start()

chunk_count = 0
try:
    for chunk in streamer:
        if chunk is None or len(chunk) == 0:
            continue
        
        chunk_count += 1
        print(f" -> Received chunk {chunk_count} via streamer at {time.time()-t0:.2f}s")

        # Convert tensor/array to standard numpy structure
        audio_chunk = chunk.cpu().numpy() if torch.is_tensor(chunk) else np.asarray(chunk)
        
        # Normalize amplitude to prevent digital clipping
        peak = np.max(np.abs(audio_chunk))
        if peak == 0: peak = 1.0
        audio_chunk = (audio_chunk / peak) * 0.9

        # Convert float32 array (-1.0 to 1.0) into signed 16-bit integers
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Write raw PCM bytes directly to the file payload
        wav_file.writeframes(audio_int16.tobytes())
finally:
    gen_thread.join()
    wav_file.close()

print(f"\n✅ Finished! Audio chunks combined and exported to: {os.path.abspath(OUTPUT_WAV_FILENAME)}")