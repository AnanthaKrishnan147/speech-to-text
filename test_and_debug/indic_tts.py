"""
Low-latency local streaming voice assistant.
Gemini (streaming text) -> Indic Parler-TTS (streaming audio) -> sounddevice (direct speaker output)

No wav files. No browser bridge. No Colab. Audio is pushed straight to your
system's audio output device as it's generated.

Run this on a machine with a GPU for best results (CUDA). CPU will work but
will be noticeably slower per-chunk since Parler-TTS is not a tiny model.

Install:
    pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the right CUDA build for your GPU
    pip install parler-tts sounddevice numpy transformers google-genai

    # parler-tts (AI4Bharat fork) usually needs to be installed from git:
    pip install git+https://github.com/huggingface/parler-tts.git
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
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
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
LLM_MODEL = "gpt-4o-mini"  # swap for whichever chat model you have access to

# Lower play_steps = lower first-chunk latency, at the cost of more per-chunk
# python/GPU call overhead. 15-30 is a good real-time range.
# (800, from the original code, is why you were waiting 10-15s per sentence.)
PLAY_STEPS = 30

# How many seconds of audio to buffer up before playback starts. This is the
# "4-5s initial delay is fine" budget you mentioned -- it exists specifically
# so that if generation briefly falls behind real-time later on, there's a
# cushion of already-generated audio to play from instead of dead air.
PREBUFFER_SECONDS = 3.0

LANGUAGE_TTS_CONFIG = {
    "EN": {"speaker_desc": "Thoma speaks in a Conversation style with a warm, natural tone, at a relaxed conversational pace, sounding unscripted. The recording is very clear audio, close-sounding, with no background noise."},
    "ML": {"speaker_desc": "Anjali speaks in a Conversation style with a warm, natural, and expressive tone, at a normal, relaxed pace as if speaking casually to a friend. Her voice has subtle emotional depth and natural intonation, sounding unscripted. The recording is very clear audio, close-sounding, with no background noise or reverberation."},
    "TA": {"speaker_desc": "Jaya speaks in a Conversation style with a warm, natural tone, at a relaxed conversational pace, sounding unscripted rather than read aloud. The recording is very clear audio, close-sounding, with no background noise."},
    "HI": {"speaker_desc": "Divya speaks in a Conversation style with a warm, natural, expressive tone, at a relaxed conversational pace, sounding unscripted. The recording is very clear audio, close-sounding, with no background noise or reverberation."},
}
DEFAULT_LANG = "EN"
LANGUAGE_NAMES = {"EN": "English", "HI": "Hindi", "ML": "Malayalam", "TA": "Tamil"}

# --------------------------------------------------------------------------
# 3. LOAD MODEL ONCE
# --------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

print(f"[tts] Loading Indic Parler-TTS on {device}...")
repo_id = "ai4bharat/indic-parler-tts"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)
tts_model.eval()
tts_tokenizer = AutoTokenizer.from_pretrained(repo_id)
tts_description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
sample_rate = tts_model.config.sampling_rate

_description_cache = {}
for lang, cfg in LANGUAGE_TTS_CONFIG.items():
    _description_cache[lang] = tts_description_tokenizer(cfg["speaker_desc"], return_tensors="pt").to(device)

tts_lock = threading.Lock()

# --------------------------------------------------------------------------
# 4. DIRECT AUDIO OUTPUT (no wav file, no browser — straight to sound card)
# --------------------------------------------------------------------------
audio_out_queue = queue.Queue()
_stream_stop = threading.Event()

def audio_playback_worker():
    """Pulls PCM float32 chunks off a queue and writes them straight to the
    output device using a single persistent OutputStream.

    Key fix: this does NOT start writing the instant the first chunk shows up.
    It first accumulates PREBUFFER_SECONDS worth of audio, then flushes that
    cushion and switches to normal streaming. This is exactly what a video
    player's "buffering..." spinner does -- it trades a small fixed startup
    delay for immunity to short-term generation slowdowns later on, which is
    what was causing the choppy/gappy playback.

    If the queue ever empties out mid-playback (generation fell behind for a
    sustained stretch, not just briefly), it re-buffers rather than playing
    silence or stuttering chunk-by-chunk.
    """
    prebuffer_samples = int(PREBUFFER_SECONDS * sample_rate)

    # latency='high' tells PortAudio to use a larger internal buffer, which
    # further smooths over small timing jitter between our writes.
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", latency="high") as stream:
        pending = []
        pending_samples = 0
        buffered_up = False

        while not _stream_stop.is_set():
            try:
                chunk = audio_out_queue.get(timeout=0.5)
            except queue.Empty:
                # Queue ran dry. If we were mid-stream, drop back into
                # buffering mode instead of stalling on writes one at a time.
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
                # else: keep accumulating, don't write yet
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

    # Very short fragments (a few characters/one word) can produce so few
    # codec tokens that the streamer's final delay-pattern flush ends up with
    # a zero-length tensor and crashes DAC's decoder. Skip fragments below a
    # safe length instead of letting the whole worker thread die.
    MIN_CHARS_FOR_TTS = 4
    if len(text) < MIN_CHARS_FOR_TTS:
        print(f"[tts] skipping too-short fragment: {text!r}")
        return

    lang_tag = lang_tag.upper() if lang_tag.upper() in LANGUAGE_TTS_CONFIG else DEFAULT_LANG
    description_inputs = _description_cache[lang_tag]
    prompt_inputs = tts_tokenizer(text, return_tensors="pt").to(device)

    streamer = ParlerTTSStreamer(tts_model, device=device, play_steps=PLAY_STEPS)

    generation_kwargs = dict(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )

    def _run_generate():
        try:
            tts_model.generate(**generation_kwargs)
        except RuntimeError as e:
            # Catches the "Kernel size can't be greater than actual input
            # size" crash from an edge-case tiny final chunk, so it doesn't
            # take down the whole worker thread.
            print(f"[tts] generation error for phrase {text!r}: {e}")

    gen_thread = threading.Thread(target=_run_generate)

    with tts_lock:
        t0 = time.time()
        first_chunk = True
        gen_thread.start()
        try:
            for chunk in streamer:
                if chunk is None or len(chunk) == 0:
                    continue
                if first_chunk:
                    print(f"[tts] first audio chunk in {time.time()-t0:.2f}s", flush=True)
                    first_chunk = False

                audio_chunk = chunk.cpu().numpy() if torch.is_tensor(chunk) else np.asarray(chunk)
                peak = np.max(np.abs(audio_chunk))
                if peak == 0:
                    peak = 1.0
                audio_chunk = (audio_chunk / peak) * 0.9

                # push straight to the playback thread -- no eval_js, no browser, no file
                audio_out_queue.put(audio_chunk.astype(np.float32))
        except RuntimeError as e:
            print(f"[tts] streamer error for phrase {text!r}: {e}")

        gen_thread.join()

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
conversation_history = []  # list of {"role": ..., "content": ...} dicts
MAX_HISTORY_MESSAGES = 12

# Sentence-level only. Splitting on commas too (as before) meant more, smaller
# TTS calls -- each call has fixed startup overhead, so more calls = more total
# dead-air risk between chunks. Since you can tolerate 4-5s of initial latency,
# fewer/bigger calls with a healthy playback buffer (below) is more stable.
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
    print("\n⚡ Local low-latency voice assistant ready.")
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