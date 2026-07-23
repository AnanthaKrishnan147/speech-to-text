import re
import sys
import subprocess
import threading
import queue

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
TRANSCRIBER_SCRIPT_PATH = "main.py"
PYTHON_EXECUTABLE = sys.executable

LLM_MODEL = "gpt-4o"
LLM_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep replies short and conversational "
    "since they will be read aloud. Reply in the same language the user spoke in."
)

# Matches lines like "[EN 95%] FORCED FINAL (max hold): Hello, can you hear me?"
# Captures the language tag (EN/ML/TA/HI...) and the transcript after the last colon.
FINAL_LINE_PATTERN = re.compile(r"\[(\w{2,3})[^\]]*\].*FINAL[^:]*:\s*(.+)$")
# Fallback for plain "FINAL: text" lines without a language tag
PLAIN_FINAL_PATTERN = re.compile(r"FINAL:\s*(.+)$")

# --------------------------------------------------------------------------
# LANGUAGE -> TTS DESCRIPTION MAP
# Tune speaker names / phrasing per language. Speakers must exist for that
# language in ai4bharat/indic-parler-tts (see model card speaker table).
# --------------------------------------------------------------------------
LANGUAGE_TTS_CONFIG = {
    "EN": {
        "speaker_desc": (
            "Thoma speaks in a Conversation style with a warm, natural tone, "
            "at a relaxed conversational pace, sounding unscripted. "
            "The recording is very clear audio, close-sounding, with no background noise."
        ),
    },
    "ML": {
        "speaker_desc": (
            "Anjali speaks in a Conversation style with a warm, natural, and expressive tone, "
            "at a normal, relaxed pace as if speaking casually to a friend. Her voice has "
            "subtle emotional depth and natural intonation, sounding unscripted. "
            "The recording is very clear audio, close-sounding, with no background noise or reverberation."
        ),
    },
    "TA": {
        "speaker_desc": (
            "Jaya speaks in a Conversation style with a warm, natural tone, "
            "at a relaxed conversational pace, sounding unscripted rather than read aloud. "
            "The recording is very clear audio, close-sounding, with no background noise."
        ),
    },
    "HI": {
        "speaker_desc": (
            "Divya speaks in a Conversation style with a warm, natural, expressive tone, "
            "at a relaxed conversational pace, sounding unscripted. "
            "The recording is very clear audio, close-sounding, with no background noise or reverberation."
        ),
    },
}
DEFAULT_LANG = "EN"  # fallback if no language tag is detected

# --------------------------------------------------------------------------
# INDIC PARLER-TTS SETUP
# --------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

print(f"[tts] Loading Indic Parler-TTS on {device}...")
repo_id = "ai4bharat/indic-parler-tts"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(repo_id)
tts_description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
sample_rate = tts_model.config.sampling_rate

# Pre-tokenize each language's description once (reused every call)
_description_cache = {}
for lang, cfg in LANGUAGE_TTS_CONFIG.items():
    _description_cache[lang] = tts_description_tokenizer(
        cfg["speaker_desc"], return_tensors="pt"
    ).to(device)

tts_lock = threading.Lock()  # serialize playback so replies don't overlap


def speak_streaming(text: str, lang_tag: str = DEFAULT_LANG):
    if not text.strip():
        return

    lang_tag = lang_tag.upper() if lang_tag.upper() in LANGUAGE_TTS_CONFIG else DEFAULT_LANG
    description_inputs = _description_cache[lang_tag]
    prompt_inputs = tts_tokenizer(text, return_tensors="pt").to(device)

    streamer = ParlerTTSStreamer(tts_model, device=device, play_steps=int(sample_rate * 0.5))

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

    gen_thread = threading.Thread(target=tts_model.generate, kwargs=generation_kwargs)

    with tts_lock:
        gen_thread.start()
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
        stream.start()
        
        try:
            for chunk in streamer:
                if chunk is None or len(chunk) == 0:
                    continue
                    
                audio_chunk = chunk.numpy() if torch.is_tensor(chunk) else np.asarray(chunk)
                
                # BUG FIX: Squeeze array to 1D so sounddevice accepts it correctly
                audio_chunk = np.squeeze(audio_chunk).astype(np.float32)
                
                # BUG FIX: Removed dynamic per-chunk normalization to prevent audio popping.
                # (Optional) If it is consistently too quiet, you can apply a static gain here:
                # audio_chunk = audio_chunk * 1.2 
                
                stream.write(audio_chunk)
        finally:
            stream.stop()
            stream.close()
            gen_thread.join()


# --------------------------------------------------------------------------
# LLM CLIENT
# --------------------------------------------------------------------------
llm_client = OpenAI()

conversation_history = []
MAX_HISTORY_MESSAGES = 12


def get_llm_reply(user_text: str) -> str:
    global conversation_history # Ensure we modify the global list
    
    conversation_history.append({"role": "user", "content": user_text})
    
    # BUG FIX: Actually trim the history in memory to prevent indefinite growth
    conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}] + conversation_history

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=messages,
    )

    reply_text = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply_text})
    return reply_text


# --------------------------------------------------------------------------
# SUBPROCESS STDOUT READER
# --------------------------------------------------------------------------
def stream_transcriber_output(proc: subprocess.Popen, line_queue: queue.Queue):
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        print(f"[transcriber] {line}")

        match = FINAL_LINE_PATTERN.search(line)
        if match:
            lang_tag, transcript = match.group(1), match.group(2).strip()
            if transcript:
                line_queue.put((lang_tag, transcript))
            continue

        # fallback if no language tag was present in the line
        plain_match = PLAIN_FINAL_PATTERN.search(line)
        if plain_match:
            transcript = plain_match.group(1).strip()
            if transcript:
                line_queue.put((DEFAULT_LANG, transcript))


def main():
    print(f"Launching transcriber subprocess: {PYTHON_EXECUTABLE} {TRANSCRIBER_SCRIPT_PATH}")

    proc = subprocess.Popen(
        [PYTHON_EXECUTABLE, "-u", TRANSCRIBER_SCRIPT_PATH], 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    transcript_queue = queue.Queue()
    reader_thread = threading.Thread(
        target=stream_transcriber_output, args=(proc, transcript_queue), daemon=True
    )
    reader_thread.start()

    print("Bridge ready. Waiting for finalized transcripts from the subprocess...\n")

    try:
        while True:
            # BUG FIX: Check if the subprocess died to prevent a deadlock
            if proc.poll() is not None:
                print("\nTranscriber subprocess exited unexpectedly. Shutting down bridge.")
                break
                
            try:
                # BUG FIX: Add a timeout so the loop can check proc.poll() periodically
                lang_tag, transcript = transcript_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            print(f"\n>>> USER [{lang_tag}]: {transcript}")

            try:
                reply = get_llm_reply(transcript)
            except Exception as e:
                print(f"[llm error] {e}")
                continue

            print(f">>> ASSISTANT: {reply}\n")
            speak_streaming(reply, lang_tag=lang_tag)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        proc.terminate()
        proc.wait() # BUG FIX: Wait for the subprocess to clean up to prevent zombies


if __name__ == "__main__":
    main()