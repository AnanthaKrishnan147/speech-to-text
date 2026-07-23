"""
Ultra-low latency streaming voice assistant.
OpenAI (streaming text) -> Meta MMS TTS (Pre-loaded VITS models) -> sounddevice (direct speaker output)

All models are eagerly loaded into memory at initialization to guarantee zero-latency language switching.
"""

import os
import re
import sys
import queue
import threading
import time
import numpy as np
import torch
import sounddevice as sd
from openai import OpenAI
from transformers import VitsModel, AutoTokenizer
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 1. API KEY (from .env file in the same directory as this script)
# --------------------------------------------------------------------------
load_dotenv()  # reads a .env file and populates os.environ

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Create a .env file next to this script "
        "containing a line like:\nOPENAI_API_KEY=sk-...your-key-here..."
    )

# --------------------------------------------------------------------------
# 2. CONFIG
# --------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"  
PREBUFFER_SECONDS = 3.0

# Meta MMS sub-repositories per language
LANGUAGE_MMS_REPOS = {
    "EN": "facebook/mms-tts-eng",  # English
    "ML": "facebook/mms-tts-mal",  # Malayalam
    "TA": "facebook/mms-tts-tam",  # Tamil
    "HI": "facebook/mms-tts-hin",  # Hindi
}
DEFAULT_LANG = "EN"
LANGUAGE_NAMES = {"EN": "English", "HI": "Hindi", "ML": "Malayalam", "TA": "Tamil"}

# --------------------------------------------------------------------------
# 3. EAGER LOAD ALL MODELS AT STARTUP (For Ultra-Low Latency)
# --------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sample_rate = 16000  

model_cache = {}
tokenizer_cache = {}
tts_lock = threading.Lock()

print(f"\n[tts] Initializing Ultra-Low Latency MMS Manager on {device}...")
print("[tts] Pre-loading ALL language models into memory. Please wait...")

# Eagerly load all models right now so they never load during runtime
for lang, repo_id in LANGUAGE_MMS_REPOS.items():
    t_start = time.time()
    print(f" -> Loading {LANGUAGE_NAMES[lang]} ({repo_id})...", end="", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = VitsModel.from_pretrained(repo_id).to(device)
    model.eval()
    
    model_cache[lang] = model
    tokenizer_cache[lang] = tokenizer
    print(f" Done in {time.time() - t_start:.2f}s")

print("[tts] All models successfully cached in memory. Ready for instant switching!\n")

def get_mms_model(lang_tag: str):
    """Instantly returns the pre-loaded model from memory cache."""
    lang_tag = lang_tag.upper() if lang_tag.upper() in model_cache else DEFAULT_LANG
    return model_cache[lang_tag], tokenizer_cache[lang_tag]

# --------------------------------------------------------------------------
# 4. DIRECT AUDIO OUTPUT
# --------------------------------------------------------------------------
audio_out_queue = queue.Queue()
_stream_stop = threading.Event()

def audio_playback_worker():
    prebuffer_samples = int(PREBUFFER_SECONDS * sample_rate)

    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", latency="high") as stream:
        pending = []
        pending_samples = 0
        buffered_up = False

        while not _stream_stop.is_set():
            try:
                chunk = audio_out_queue.get(timeout=0.5)
            except queue.Empty:
                if buffered_up and pending_samples == 0:
                    buffered_up = False
                continue
            if chunk is None:
                continue

            pending.append(chunk)
            pending_samples += len(chunk)

            if not buffered_up:
                if pending_samples >= prebuffer_samples:
                    stream.write(np.concatenate(pending))
                    pending, pending_samples = [], 0
                    buffered_up = True
            else:
                stream.write(chunk)
                pending, pending_samples = [], 0

playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
playback_thread.start()

# --------------------------------------------------------------------------
# 5. TTS GENERATION -> STRAIGHT TO AUDIO QUEUE
# --------------------------------------------------------------------------
def generate_audio_for_phrase(text: str, lang_tag: str):
    text = text.strip()
    if not text:
        return

    MIN_CHARS_FOR_TTS = 4
    if len(text) < MIN_CHARS_FOR_TTS:
        print(f"[tts] skipping too-short fragment: {text!r}")
        return

    try:
        with tts_lock:
            t0 = time.time()
            model, tokenizer = get_mms_model(lang_tag)
            
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                audio_data = outputs.waveform[0].cpu().numpy()
            
            print(f" [audio generated in {time.time()-t0:.2f}s]", end="", flush=True)

            peak = np.max(np.abs(audio_data))
            if peak == 0:
                peak = 1.0
            audio_chunk = (audio_data / peak) * 0.9

            audio_out_queue.put(audio_chunk.astype(np.float32))
            
    except Exception as e:
        print(f"\n[tts] generation error for phrase {text!r}: {e}")

def tts_worker(sentence_queue, lang_tag):
    while True:
        phrase = sentence_queue.get()
        if phrase is None:
            break
        generate_audio_for_phrase(phrase, lang_tag)
        sentence_queue.task_done()

# --------------------------------------------------------------------------
# 6. OPENAI STREAMING
# --------------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
conversation_history = []  
MAX_HISTORY_MESSAGES = 12
_SPLIT_RE = re.compile(r'([.!?\n])')

def process_and_stream_reply(user_text: str, lang_tag: str) -> str:
    global conversation_history

    conversation_history.append({"role": "user", "content": user_text})
    trimmed_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    target_lang_name = LANGUAGE_NAMES.get(lang_tag, "English")
    dynamic_system_prompt = (
        f"You are a helpful voice assistant. Keep replies short and conversational. "
        f"You MUST reply entirely in {target_lang_name} language script."
    )

    messages = [{"role": "system", "content": dynamic_system_prompt}] + trimmed_history

    sentence_queue = queue.Queue()
    audio_thread = threading.Thread(target=tts_worker, args=(sentence_queue, lang_tag))
    audio_thread.start()

    print(f"\n>>> ASSISTANT [{lang_tag}]: ", end="")
    response_stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=800,
        stream=True,
    )

    full_response = ""
    buffer = ""

    for chunk in response_stream:
        delta = chunk.choices[0].delta
        text_chunk = delta.content if delta and delta.content else ""
        if not text_chunk:
            continue
        sys.stdout.write(text_chunk)
        sys.stdout.flush()
        full_response += text_chunk
        buffer += text_chunk

        if _SPLIT_RE.search(buffer):
            parts = _SPLIT_RE.split(buffer)
            for i in range(0, len(parts) - 1, 2):
                phrase = parts[i] + parts[i + 1]
                if phrase.strip():
                    sentence_queue.put(phrase.strip())
            buffer = parts[-1]

    if buffer.strip():
        sentence_queue.put(buffer.strip())

    print("\n")
    sentence_queue.put(None)
    audio_thread.join()

    conversation_history.append({"role": "assistant", "content": full_response})
    return full_response

def main():
    print("⚡ Ultra low-latency voice assistant engine running.")
    try:
        while True:
            user_input = input(">>> You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                break

            lang_tag = DEFAULT_LANG
            transcript = user_input

            match = re.match(r"^\[(\w{2,3})\]\s*(.*)", user_input)
            if match:
                lang_tag = match.group(1).upper()
                transcript = match.group(2).strip()
            try:
                process_and_stream_reply(transcript, lang_tag)
            except Exception as e:
                print(f"\n[Error] {e}")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        _stream_stop.set()

if __name__ == "__main__":
    main()