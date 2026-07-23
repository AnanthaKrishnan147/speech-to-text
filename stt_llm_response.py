"""
llm_bridge.py

Runs the real-time STT pipeline (stt_realtime.py) and, for every finalized
transcript it produces, sends the text to an OpenAI LLM and prints the reply.

It also logs a debug line per turn measuring the delay between the moment the
user stopped speaking (the same `speech_end_time` reference the STT script
uses for its own timing_debug.log) and the moment the LLM's response comes
back. This is written to llm_timing_debug.log in the same style/format as the
STT script's own debug log.

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."

Run:
    python llm_bridge.py

Make sure stt_realtime.py (the modified STT file) is in the same directory
(or importable on PYTHONPATH) before running this.
"""

import os
import time
import threading
import queue
import traceback

from dotenv import load_dotenv
load_dotenv()  # reads a .env file in the current working directory into os.environ

from openai import OpenAI

import main as stt

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. The user is speaking to you "
    "via a Malayalam-English code-switched speech recognition pipeline, so "
    "their transcribed text may mix both languages or contain minor ASR "
    "errors. Reply naturally and briefly."
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------
# Debug logging: speech-end -> LLM-response delay
# ---------------------------------------------------------------
LLM_DEBUG_LOG_PATH = "llm_timing_debug.log"
_llm_debug_log_lock = threading.Lock()


def log_llm_timing_debug(utt_id, user_text, llm_text, total_delay_sec, llm_call_sec, tag=""):
    """
    Append one line per LLM turn:
      - total_delay_sec: time from the user finishing speaking to the LLM
        response being ready (this is the number you asked for).
      - llm_call_sec: time spent purely inside the OpenAI API call, broken
        out separately so you can tell API latency apart from any queueing/
        ASR-finalization delay upstream.
    """
    line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] utt#{utt_id} "
        f"speech_to_response={total_delay_sec*1000:.0f}ms "
        f"llm_call={llm_call_sec*1000:.0f}ms {tag} "
        f"user=\"{user_text}\" -> llm=\"{llm_text}\"\n"
    )
    with _llm_debug_log_lock:
        with open(LLM_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------
# Bridge: STT transcript -> LLM -> printed reply
# ---------------------------------------------------------------
transcript_queue = queue.Queue()

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]
_history_lock = threading.Lock()

MAX_HISTORY_MESSAGES = 20  # keep the last N messages (excluding system prompt) to bound context/cost


def on_final_transcript(utt_id, transcript, lang, confidence, speech_end_time):
    """Registered with stt_realtime; called from the STT worker thread whenever
    a transcript is finalized. Just hands it off to the LLM worker queue so we
    never block the STT pipeline."""
    print(f"[bridge] received final transcript utt#{utt_id}: \"{transcript}\" -> queuing for LLM")
    transcript_queue.put((utt_id, transcript, lang, confidence, speech_end_time))


def _trim_history():
    with _history_lock:
        if len(conversation_history) > MAX_HISTORY_MESSAGES + 1:
            # keep system prompt (index 0) + most recent messages
            del conversation_history[1:len(conversation_history) - MAX_HISTORY_MESSAGES]


def handle_transcript(utt_id, transcript, lang, confidence, speech_end_time):
    if not transcript.strip():
        return

    with _history_lock:
        conversation_history.append({"role": "user", "content": transcript})
        messages_snapshot = list(conversation_history)

    call_start = time.time()
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages_snapshot,
        )
        llm_text = response.choices[0].message.content.strip()
    except Exception as e:
        call_end = time.time()
        print(f"[LLM ERROR utt#{utt_id}] {e}")
        log_llm_timing_debug(
            utt_id, transcript, f"<error: {e}>",
            call_end - speech_end_time, call_end - call_start, tag="[Error]"
        )
        return
    call_end = time.time()

    with _history_lock:
        conversation_history.append({"role": "assistant", "content": llm_text})
    _trim_history()

    total_delay = call_end - speech_end_time
    llm_call_duration = call_end - call_start

    print(f"[{lang.upper()} {confidence:.0f}%] utt#{utt_id} User: {transcript}")
    print(f"[LLM] utt#{utt_id} Assistant: {llm_text}")

    log_llm_timing_debug(utt_id, transcript, llm_text, total_delay, llm_call_duration, tag="[Final]")


def llm_worker():
    while True:
        utt_id, transcript, lang, confidence, speech_end_time = transcript_queue.get()
        try:
            handle_transcript(utt_id, transcript, lang, confidence, speech_end_time)
        except Exception:
            print("[llm_worker error]")
            traceback.print_exc()


def main():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(
            "WARNING: OPENAI_API_KEY is not set (checked os.environ after load_dotenv()).\n"
            "  - Confirm .env is in the SAME directory you're running `python llm_bridge.py` from.\n"
            "  - Confirm the line reads exactly: OPENAI_API_KEY=sk-...  (no quotes, no spaces around =)\n"
            "  - Every LLM call below will fail until this is fixed."
        )
    else:
        print(f"[bridge] OPENAI_API_KEY loaded: {key[:7]}...{key[-4:]} (len={len(key)})")

    print(f"[bridge] using model: {OPENAI_MODEL}")
    print("[bridge] registering transcript callback with STT pipeline...")
    stt.register_transcript_callback(on_final_transcript)
    print("[bridge] callback registered. Starting LLM worker thread...")

    worker_thread = threading.Thread(target=llm_worker, daemon=True)
    worker_thread.start()

    # This blocks (loads all STT models, opens the mic stream, waits for
    # Enter) exactly like running stt_realtime.py directly would.
    stt.start_listening()


if __name__ == "__main__":
    main()