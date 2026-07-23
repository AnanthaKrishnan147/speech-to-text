import os
import json
import asyncio
import threading
import queue
import traceback
import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import sounddevice as sd

import nemo.collections.asr as nemo_asr

import time

# --- NEW: Import faster-whisper instead of whisper ---
from faster_whisper import WhisperModel

DEBUG_LOG_PATH = "timing_debug.log"
_debug_log_lock = threading.Lock()

def log_timing_debug(utt_id, transcript, lang, confidence, delay_sec, tag=""):
    """Append one line per generated transcript: delay since speech stopped + language."""
    line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] utt#{utt_id} "
        f"lang={lang} conf={confidence:.0f}% "
        f"delay={delay_sec*1000:.0f}ms {tag} "
        f"text=\"{transcript}\"\n"
    )
    with _debug_log_lock:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)

# ---------------------------------------------------------------
# Transcript callback hook
# ---------------------------------------------------------------
# Any external module (e.g. an LLM bridge) can register a function here to be
# notified every time a FINAL transcript is produced. The callback receives:
#   (utt_id, transcript, lang, confidence, speech_end_time)
# speech_end_time is a time.time() timestamp marking the moment the VAD judged
# the user had stopped speaking (same reference point used for timing_debug.log).
TRANSCRIPT_CALLBACKS = []

def register_transcript_callback(fn):
    """Register a callable to be invoked on every finalized transcript."""
    TRANSCRIPT_CALLBACKS.append(fn)

def _emit_transcript(utt_id, transcript, lang, confidence, speech_end_time):
    for cb in TRANSCRIPT_CALLBACKS:
        try:
            cb(utt_id, transcript, lang, confidence, speech_end_time)
        except Exception:
            print("[transcript callback error]")
            traceback.print_exc()

# ---------------------------------------------------------------
# PyTorch JIT & Device Configuration
# ---------------------------------------------------------------
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on hardware device: {device}\n")

# ---------------------------------------------------------------
# NeMo FastConformer LID (replaces SpeechBrain VoxLingua107)
# ---------------------------------------------------------------
FASTCONFORMER_MODEL_NAME = "stt_en_fastconformer_ctc_large"  # swap for your fine-tuned .nemo checkpoint later
LID_LABELS = ["en", "ml", "other"]                            # frame-level class labels
LID_HEAD_WEIGHTS_PATH = "frame_lang_head.pt"                  # set once you've trained your head

# Languages this pipeline actually needs to distinguish between.
ALLOWED_LANGUAGE_CODES = {
    "en",  # English -> routed to Whisper transcription
    "hi", "ta", "te", "kn", "ml", "mr", "gu", "pa", "bn", "as",  # -> routed to IndicConformer
}


class FrameLanguageHead(torch.nn.Module):
    """Lightweight per-frame language classifier on top of FastConformer encoder outputs."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (batch, time, input_dim) -> (batch, time, num_classes)
        return self.net(x)


print("Loading NeMo FastConformer encoder for LID...")
_lid_asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=FASTCONFORMER_MODEL_NAME)
_lid_asr_model = _lid_asr_model.to(device)
_lid_asr_model.eval()

_lid_encoder = _lid_asr_model.encoder
_lid_preprocessor = _lid_asr_model.preprocessor
_lid_encoder_dim = _lid_encoder.d_model if hasattr(_lid_encoder, "d_model") else 512

lid_head = FrameLanguageHead(input_dim=_lid_encoder_dim, num_classes=len(LID_LABELS)).to(device)
lid_head.eval()

if os.path.exists(LID_HEAD_WEIGHTS_PATH):
    print(f"Loading trained LID head weights from {LID_HEAD_WEIGHTS_PATH}")
    lid_head.load_state_dict(torch.load(LID_HEAD_WEIGHTS_PATH, map_location=device))
else:
    print(
        f"WARNING: {LID_HEAD_WEIGHTS_PATH} not found — LID head is UNTRAINED. "
        "Predictions will be meaningless until you train and save this head."
    )


def classify_language(wav_tensor: torch.Tensor):
    USE_TRAINED_FASTCONFORMER_HEAD = os.path.exists(LID_HEAD_WEIGHTS_PATH)

    if USE_TRAINED_FASTCONFORMER_HEAD:
        length = torch.tensor([wav_tensor.shape[1]], device=device)
        with torch.no_grad():
            processed_signal, processed_len = _lid_preprocessor(
                input_signal=wav_tensor, length=length
            )
            encoded, encoded_len = _lid_encoder(
                audio_signal=processed_signal, length=processed_len
            )
            encoded = encoded.transpose(1, 2)
            logits = lid_head(encoded)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            frame_preds = torch.argmax(probs, dim=-1).cpu().numpy()

        frame_labels = [LID_LABELS[p] for p in frame_preds]
        if len(frame_labels) == 0:
            return "en", 0.0, frame_labels
        values, counts = np.unique(frame_preds, return_counts=True)
        majority_idx = values[np.argmax(counts)]
        majority_label = LID_LABELS[majority_idx]
        confidence = 100.0 * counts.max() / counts.sum()
        return majority_label, confidence, frame_labels

    # --- Fallback: Faster-Whisper's language detector ---
    audio_np = wav_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    try:
        # faster-whisper returns the top language, top probability, and a list of all language probabilities
        _, _, all_language_probs = whisper_model.detect_language(audio_np)
    except ValueError as e:
        raise RuntimeError(
            "whisper_model is an English-only checkpoint and cannot detect "
            "language. Load a multilingual Whisper checkpoint (e.g. "
            "'base') instead of an '.en' variant."
        ) from e

    # all_language_probs is a list of tuples: [("en", 0.95), ("fr", 0.01), ...]
    filtered_probs = {lang: prob for lang, prob in all_language_probs if lang in ALLOWED_LANGUAGE_CODES}

    if not filtered_probs:
        raise RuntimeError(
            "None of ALLOWED_LANGUAGE_CODES were found in Whisper's language "
            "probability output."
        )

    detected = max(filtered_probs, key=filtered_probs.get)
    total = sum(filtered_probs.values())
    confidence = (filtered_probs[detected] / total) * 100.0 if total > 0 else 0.0

    label = detected

    return label, confidence, []


# ---------------------------------------------------------------
# ASR models
# ---------------------------------------------------------------
print("Loading Faster-Whisper (English)...")
# --- NEW: Initialize Faster-Whisper with int8 CPU optimization ---
compute_type = "float16" if device == "cuda" else "int8"
whisper_model = WhisperModel("base", device=device, compute_type=compute_type)

print("Loading AI4Bharat IndicConformer-600M...")
indic_model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
).to(device)
indic_model.eval()

VOXLINGUA_TO_INDIC_MAP = {
    "hi": "hi", "ta": "ta", "te": "te", "kn": "kn",
    "ml": "ml", "mr": "mr", "gu": "gu", "pa": "pa",
    "bn": "bn", "or": "or", "as": "as"
}

def transcribe_indic_audio(wav, language_code, decoding_strategy="ctc"):
    wav = wav.to(device)
    with torch.no_grad():
        out = indic_model(wav, language_code, decoding_strategy)
    return out[0] if isinstance(out, (list, tuple)) else out

# ---------------------------------------------------------------
# Silero VAD — streaming voice activity detection
# ---------------------------------------------------------------
print("Loading Silero VAD...")
silero_model, silero_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
silero_model.to(device)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = silero_utils

# ---------------------------------------------------------------
# Semantic turn detector
# ---------------------------------------------------------------
print("Loading LiveKit semantic turn-detector model (multilingual)...")

_TD_REPO = "livekit/turn-detector"
_TD_REVISION = "v1.2.0"
_TD_ONNX_FILE = "model_q8.onnx"

_td_model_path = hf_hub_download(
    repo_id=_TD_REPO,
    filename=_TD_ONNX_FILE,
    subfolder="onnx",
    revision=_TD_REVISION,
)
_td_tokenizer = AutoTokenizer.from_pretrained(
    _TD_REPO, revision=_TD_REVISION, truncation_side="left"
)
_td_session = ort.InferenceSession(_td_model_path, providers=["CPUExecutionProvider"])
_td_input_names = [inp.name for inp in _td_session.get_inputs()]
print(f"[turn-detector] ONNX inputs: {_td_input_names}")

class SemanticTurnDetector:
    MAX_HISTORY_TURNS = 6
    MAX_TOKENS = 128

    def __init__(self, tokenizer, session, threshold=0.5):
        self.tokenizer = tokenizer
        self.session = session
        self.threshold = threshold
        self.eou_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.input_names = [inp.name for inp in session.get_inputs()]

    def _format(self, history):
        text = self.tokenizer.apply_chat_template(
            history[-self.MAX_HISTORY_TURNS:],
            tokenize=False,
            add_generation_prompt=False,
        )
        if text.endswith("<|im_end|>\n"):
            text = text[: -len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")]
        return text

    def predict(self, history) -> float:
        text = self._format(history)
        enc = self.tokenizer(
            text, return_tensors="np", truncation=True, max_length=self.MAX_TOKENS
        )
        full_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        ort_inputs = {k: v for k, v in full_inputs.items() if k in self.input_names}

        outputs = self.session.run(None, ort_inputs)
        logits = outputs[0]

        if logits.ndim == 1:
            val = float(logits[0])
            prob = val if 0.0 <= val <= 1.0 else 1 / (1 + np.exp(-val))
            return prob

        elif logits.ndim == 3:
            next_token_logits = logits[0, -1, :]
            probs = np.exp(next_token_logits - next_token_logits.max())
            probs = probs / probs.sum()
            return float(probs[self.eou_token_id])

        else:
            raise ValueError(f"Unexpected turn-detector output shape: {logits.shape}")

    def is_turn_complete(self, history) -> bool:
        return self.predict(history) >= self.threshold

turn_detector = SemanticTurnDetector(_td_tokenizer, _td_session, threshold=0.5)

print("\nAll models loaded (ASR + FastConformer LID + Silero VAD + semantic turn detector).\n")

# ---------------------------------------------------------------
# LiveTranscriber Class
# ---------------------------------------------------------------
SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512
SILENCE_MS_TO_ENDPOINT = 450
SILENCE_MS_MAX_HOLD = 2600
MIN_SPEECH_MS = 250

class LiveTranscriber:
    def __init__(self):
        self.vad_iterator = VADIterator(silero_model, sampling_rate=SAMPLE_RATE, threshold=0.5)
        self.pcm_buf = np.zeros(0, dtype=np.float32)
        self.frame_leftover = np.zeros(0, dtype=np.float32)

        self.speech_frames = []
        self.in_speech = False
        self.silence_ms = 0.0
        self.chat_history = []

        # Candidate-boundary bookkeeping
        self.utterance_id = 0
        self.candidate_sent_for = -1
        self.silence_since_candidate_ms = 0.0
        self.candidate_pending = False
        self.candidate_result_cache = None

        self._lock = threading.Lock()

        self.infer_q = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def push_pcm(self, float32_chunk: np.ndarray):
        try:
            self.pcm_buf = np.concatenate([self.frame_leftover, float32_chunk])
            n_frames = len(self.pcm_buf) // VAD_FRAME_SAMPLES
            usable = n_frames * VAD_FRAME_SAMPLES
            self.frame_leftover = self.pcm_buf[usable:]
            for i in range(n_frames):
                frame = self.pcm_buf[i * VAD_FRAME_SAMPLES:(i + 1) * VAD_FRAME_SAMPLES]
                self._process_frame(frame)
        except Exception:
            print("[push_pcm error]")
            traceback.print_exc()

    def _process_frame(self, frame: np.ndarray):
        frame_tensor = torch.from_numpy(frame).to(device)
        speech_dict = self.vad_iterator(frame_tensor, return_seconds=False)
        frame_ms = (VAD_FRAME_SAMPLES / SAMPLE_RATE) * 1000

        is_speech = speech_dict is not None and "start" in speech_dict
        is_end = speech_dict is not None and "end" in speech_dict

        with self._lock:
            if is_speech:
                self.in_speech = True
                self.silence_ms = 0.0
                self.silence_since_candidate_ms = 0.0
                self.candidate_pending = False
                self.candidate_sent_for = -1
                self.candidate_result_cache = None

            if self.in_speech:
                self.speech_frames.append(frame)

            if is_end:
                self.silence_ms += frame_ms
            elif self.in_speech and self._energy_is_silence(frame):
                self.silence_ms += frame_ms * 0.5
            else:
                if self.in_speech and not is_end:
                    self.silence_ms = max(0.0, self.silence_ms - frame_ms)

            if not self.in_speech:
                return

            speech_ms = len(self.speech_frames) * frame_ms

            # --- Stage 1: candidate boundary ---
            if (
                self.silence_ms >= SILENCE_MS_TO_ENDPOINT
                and speech_ms >= MIN_SPEECH_MS
                and self.candidate_sent_for != self.utterance_id
            ):
                self.candidate_sent_for = self.utterance_id
                self.candidate_pending = True
                self.silence_since_candidate_ms = self.silence_ms
                snapshot = np.concatenate(self.speech_frames)

                speech_end_time = time.time() - (self.silence_ms / 1000.0)
                self.infer_q.put(("candidate", self.utterance_id, snapshot, speech_end_time))

            # --- Safety net: force-finalize if silence keeps growing past max hold ---
            elif self.candidate_pending:
                self.silence_since_candidate_ms += frame_ms
                if self.silence_since_candidate_ms >= SILENCE_MS_MAX_HOLD:
                    self._finalize_locked()

    def _energy_is_silence(self, frame, thresh=0.01):
        return float(np.sqrt(np.mean(frame ** 2))) < thresh

    def _finalize_locked(self):
        """Must be called with self._lock held."""
        audio = np.concatenate(self.speech_frames) if self.speech_frames else np.zeros(0, dtype=np.float32)
        finished_id = self.utterance_id

        speech_end_time = time.time() - (self.silence_ms / 1000.0)

        cached = self.candidate_result_cache
        reuse_cache = (
            cached is not None
            and cached[0] == finished_id
            and len(self.speech_frames) > 0
        )

        self.speech_frames = []
        self.in_speech = False
        self.silence_ms = 0.0
        self.silence_since_candidate_ms = 0.0
        self.candidate_pending = False
        self.candidate_result_cache = None
        self.vad_iterator.reset_states()
        self.utterance_id += 1

        if reuse_cache:
            _, transcript, lang, confidence = cached
            self.chat_history = self.chat_history + [{"role": "user", "content": transcript}]
            print(f"[{lang.upper()} {confidence:.0f}%] FORCED FINAL (reused candidate, max hold): {transcript}")
            _emit_transcript(finished_id, transcript, lang, confidence, speech_end_time)
        elif len(audio) > 0:
            self.infer_q.put(("final", finished_id, audio, speech_end_time))

    def _worker_loop(self):
        while True:
            kind, utt_id, audio, speech_end_time = self.infer_q.get()
            try:
                if kind == "candidate":
                    self._handle_candidate(utt_id, audio, speech_end_time)
                else:
                    self._transcribe_and_emit(utt_id, audio, forced=True, speech_end_time=speech_end_time)
            except Exception:
                print("[worker error]")
                traceback.print_exc()

    def _handle_candidate(self, utt_id: int, audio_np: np.ndarray, speech_end_time: float):
        with self._lock:
            if utt_id != self.candidate_sent_for or utt_id != self.utterance_id:
                return

        transcript, detected_lang, confidence = self._transcribe(audio_np)

        delay_sec = time.time() - speech_end_time
        if transcript:
            log_timing_debug(utt_id, transcript, detected_lang, confidence, delay_sec, tag="[Candidate]")

        if not transcript:
            return

        with self._lock:
            self.candidate_result_cache = (utt_id, transcript, detected_lang, confidence)

        provisional_history = self.chat_history + [{"role": "user", "content": transcript}]
        eou_prob = turn_detector.predict(provisional_history)

        with self._lock:
            if utt_id != self.candidate_sent_for or utt_id != self.utterance_id:
                return

            if eou_prob >= turn_detector.threshold:
                self.chat_history = provisional_history
                self._finalize_locked()
                print(f"[{detected_lang.upper()} {confidence:.0f}%] FINAL: {transcript}")
                _emit_transcript(utt_id, transcript, detected_lang, confidence, speech_end_time)

    def _transcribe_and_emit(self, utt_id: int, audio_np: np.ndarray, forced: bool, speech_end_time: float):
        transcript, detected_lang, confidence = self._transcribe(audio_np)

        delay_sec = time.time() - speech_end_time
        tag = "[Forced Final]" if forced else "[Final]"
        if transcript:
            log_timing_debug(utt_id, transcript, detected_lang, confidence, delay_sec, tag=tag)

        if not transcript:
            return

        with self._lock:
            self.chat_history = self.chat_history + [{"role": "user", "content": transcript}]
        print(f"[{detected_lang.upper()} {confidence:.0f}%] {tag}: {transcript}")
        _emit_transcript(utt_id, transcript, detected_lang, confidence, speech_end_time)

    def _transcribe(self, audio_np: np.ndarray):
        tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)

        detected_lang, confidence, frame_labels = classify_language(tensor)

        if detected_lang == "en":
            # --- NEW: faster-whisper transcription returns a generator ---
            segments, _ = whisper_model.transcribe(
                audio_np,
                language="en",
                task="transcribe",
                beam_size=5,
            )
            transcript = " ".join([segment.text for segment in segments])
        elif detected_lang in VOXLINGUA_TO_INDIC_MAP:
            transcript = transcribe_indic_audio(tensor, VOXLINGUA_TO_INDIC_MAP[detected_lang])
        else:
            transcript = transcribe_indic_audio(tensor, "ml")

        transcript = transcript.strip()
        return transcript, detected_lang, confidence


# ---------------------------------------------------------------
# Audio Input Stream
# ---------------------------------------------------------------
BLOCK_SIZE = 2048

transcriber = None
stream = None


def start_listening():
    """
    Boots the LiveTranscriber + microphone input stream and blocks until the
    user presses Enter. Safe to call from an importing module (e.g. an LLM
    bridge) — nothing here runs automatically just from `import stt_realtime`.
    """
    global transcriber, stream

    transcriber = LiveTranscriber()
    print("Transcriber ready — waiting for audio from the mic.")

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        try:
            transcriber.push_pcm(indata[:, 0].copy())
        except Exception:
            pass

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback,
    )

    print("Listening... press Enter to stop.")
    stream.start()
    input()
    stream.stop()
    stream.close()


if __name__ == "__main__":
    start_listening()