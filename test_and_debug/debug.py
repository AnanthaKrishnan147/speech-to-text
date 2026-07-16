import os
import json
import asyncio
import threading
import queue
import traceback
import numpy as np
import torch
import torchaudio
import whisper
from transformers import AutoModel, AutoTokenizer
from speechbrain.inference.classifiers import EncoderClassifier
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import sounddevice as sd

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
# ASR / LID models
# ---------------------------------------------------------------
print("Loading SpeechBrain VoxLingua107 LID...")
lid_model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp",
    run_opts={"device": device}
)

print("Loading Whisper (English)...")
whisper_model = whisper.load_model("base", device=device)

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

class SemanticTurnDetector:
    MAX_HISTORY_TURNS = 6
    MAX_TOKENS = 128

    def __init__(self, tokenizer, session, threshold=0.5):
        self.tokenizer = tokenizer
        self.session = session
        self.threshold = threshold
        self.eou_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

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
        enc = self.tokenizer(text, return_tensors="np", truncation=True, max_length=self.MAX_TOKENS)
        ort_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(None, ort_inputs)[0]
        next_token_logits = logits[0, -1, :]
        probs = np.exp(next_token_logits - next_token_logits.max())
        probs = probs / probs.sum()
        return float(probs[self.eou_token_id])

turn_detector = SemanticTurnDetector(_td_tokenizer, _td_session, threshold=0.5)
print("\nAll models loaded.\n")

# ---------------------------------------------------------------
# LiveTranscriber Class
# ---------------------------------------------------------------
SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512          
SILENCE_MS_TO_ENDPOINT = 450     
MIN_SPEECH_MS = 250

class LiveTranscriber:
    def __init__(self):
        self.vad_iterator = VADIterator(silero_model, sampling_rate=SAMPLE_RATE, threshold=0.3) # Lowered threshold slightly
        self.pcm_buf = np.zeros(0, dtype=np.float32)
        self.frame_leftover = np.zeros(0, dtype=np.float32)

        self.speech_frames = []
        self.in_speech = False
        self.silence_ms = 0.0
        self.chat_history = []

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
            traceback.print_exc()

    def _process_frame(self, frame: np.ndarray):
        frame_tensor = torch.from_numpy(frame).to(device)
        speech_dict = self.vad_iterator(frame_tensor, return_seconds=False)
        frame_ms = (VAD_FRAME_SAMPLES / SAMPLE_RATE) * 1000

        is_speech = speech_dict is not None and "start" in speech_dict
        is_end = speech_dict is not None and "end" in speech_dict

        if is_speech:
            if not self.in_speech:
                print("\n[VAD] 🎤 Speech detected! Recording...")
            self.in_speech = True
            self.silence_ms = 0.0

        if self.in_speech:
            self.speech_frames.append(frame)

        # Adjusted energy threshold for noisy environments
        if is_end or (self.in_speech and self._energy_is_silence(frame, thresh=0.03)):
            self.silence_ms += frame_ms
        else:
            if self.in_speech:
                self.silence_ms = 0.0

        if self.in_speech and self.silence_ms >= SILENCE_MS_TO_ENDPOINT:
            speech_ms = len(self.speech_frames) * frame_ms
            if speech_ms >= MIN_SPEECH_MS:
                self._maybe_finalize()
            else:
                # False alarm (cough, mic bump)
                self.speech_frames = []
                self.in_speech = False
                self.silence_ms = 0.0
                print("[VAD] 🛑 False alarm (too short). Resetting.")

    def _energy_is_silence(self, frame, thresh=0.03):
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return rms < thresh

    def _maybe_finalize(self):
        audio = np.concatenate(self.speech_frames)
        print(f"[VAD] ⏳ End of speech detected. Queuing {len(audio)/SAMPLE_RATE:.2f}s of audio for ASR...")
        self.speech_frames = []
        self.in_speech = False
        self.silence_ms = 0.0
        self.vad_iterator.reset_states()
        self.infer_q.put(audio)

    def _worker_loop(self):
        while True:
            audio = self.infer_q.get()
            try:
                self._transcribe_and_check(audio)
            except Exception:
                traceback.print_exc()

    def _transcribe_and_check(self, audio_np: np.ndarray):
        print("[ASR] Running Language ID and Transcription...")
        tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)

        with torch.no_grad():
            out_prob, score, index, text_lab = lid_model.classify_batch(tensor)
        pred_index = int(index[0].item())
        detected_lang = text_lab[0].split(":")[0].strip().lower()
        confidence = torch.exp(out_prob[0][pred_index]).item() * 100

        if detected_lang == "en":
            tmp_path = "temp_chunk.wav"
            torchaudio.save(tmp_path, tensor.cpu(), SAMPLE_RATE)
            transcript = whisper_model.transcribe(tmp_path).get("text", "")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        elif detected_lang in VOXLINGUA_TO_INDIC_MAP:
            transcript = transcribe_indic_audio(tensor, VOXLINGUA_TO_INDIC_MAP[detected_lang])
        else:
            transcript = transcribe_indic_audio(tensor, "hi")

        transcript = transcript.strip()
        if not transcript:
            print("[ASR] Empty transcript. Ignoring.")
            return

        provisional_history = self.chat_history + [{"role": "user", "content": transcript}]
        eou_prob = turn_detector.predict(provisional_history)

        if eou_prob >= turn_detector.threshold:
            self.chat_history = provisional_history
            print(f">>> [{detected_lang.upper()} {confidence:.0f}% | EOU {eou_prob:.2f}] FINAL: {transcript}")
        else:
            self.chat_history = provisional_history
            print(f">>> [{detected_lang.upper()} {confidence:.0f}% | EOU {eou_prob:.2f}] (continuing…): {transcript}")

transcriber = LiveTranscriber()
print("Transcriber ready.")

# ---------------------------------------------------------------
# Audio Input Stream
# ---------------------------------------------------------------
BLOCK_SIZE = 2048  

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Stream Status: {status}")
    
    # Print a basic volume meter to ensure mic is working
    volume_norm = np.linalg.norm(indata) * 10
    print(f"\rMic Volume: {'|' * int(volume_norm)}", end="", flush=True)
    
    try:
        transcriber.push_pcm(indata[:, 0].copy())
    except Exception:
        traceback.print_exc() 

print("\nListing available audio devices:")
print(sd.query_devices())
print("\nListening... press Enter to stop.")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    dtype="float32",
    callback=audio_callback,
)

stream.start()
input()
stream.stop()
stream.close()