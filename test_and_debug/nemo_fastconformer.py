import queue
import threading
import time
 
import numpy as np
import torch
import pyaudio
 
import nemo.collections.asr as nemo_asr
 
 
# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000          # NeMo FastConformer models expect 16kHz mono audio
CHUNK_DURATION_SEC = 0.5     # how much audio we grab from mic per read
BUFFER_DURATION_SEC = 2.0    # sliding window length fed to the encoder
CHANNELS = 1
FORMAT = pyaudio.paInt16
 
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION_SEC)
 
# Pick any NeMo FastConformer-based model. This one is a small, fast, English
# CTC model — its encoder is the same FastConformer architecture used by
# Parakeet, so it's a good starting checkpoint to fine-tune from, or to use
# as-is to inspect the embedding pipeline. Swap for whichever pretrained
# checkpoint you plan to fine-tune (e.g. a NeMo multilingual FastConformer,
# or your own fine-tuned .nemo file once you have one).
PRETRAINED_MODEL_NAME = "stt_en_fastconformer_ctc_large"
 
# Number of language classes for your custom head (e.g. Malayalam, English, silence)
NUM_LANGUAGE_CLASSES = 3
LABELS = ["malayalam", "english", "silence_or_other"]
 
 
# --------------------------------------------------------------------------
# CUSTOM FRAME-LEVEL CLASSIFICATION HEAD
# --------------------------------------------------------------------------
class FrameLanguageHead(torch.nn.Module):
    """
    Lightweight classification head on top of FastConformer encoder outputs.
    Encoder output shape: (batch, time, feature_dim). feature_dim is 512 for
    most FastConformer variants (check encoder.d_model to confirm for your model).
    """
 
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
 
 
# --------------------------------------------------------------------------
# MICROPHONE STREAMER
# --------------------------------------------------------------------------
class MicrophoneStreamer:
    """Continuously reads audio from the microphone into a thread-safe queue."""
 
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self._stop_flag = threading.Event()
 
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
 
    def _callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
 
    def start(self):
        self._stream.start_stream()
        print("Microphone streaming started. Speak now. Ctrl+C to stop.")
 
    def stop(self):
        self._stop_flag.set()
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
 
    def read_chunk(self, timeout=1.0):
        try:
            data = self.audio_queue.get(timeout=timeout)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
        except queue.Empty:
            return None
 
 
# --------------------------------------------------------------------------
# MAIN REAL-TIME LOOP
# --------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NeMo FastConformer model on {device} ...")
 
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=PRETRAINED_MODEL_NAME
    )
    asr_model = asr_model.to(device)
    asr_model.eval()
 
    encoder = asr_model.encoder  # the FastConformer encoder submodule
    preprocessor = asr_model.preprocessor  # converts raw waveform -> log-mel features
 
    # Infer encoder output feature dim (usually 512 or 1024 depending on model size)
    encoder_dim = encoder.d_model if hasattr(encoder, "d_model") else 512
    print(f"Encoder output dimension: {encoder_dim}")
 
    lang_head = FrameLanguageHead(input_dim=encoder_dim, num_classes=NUM_LANGUAGE_CLASSES)
    lang_head = lang_head.to(device)
    lang_head.eval()
 
    # Once you've trained your head on labeled code-switch data, load it here:
    # lang_head.load_state_dict(torch.load("frame_lang_head.pt", map_location=device))
 
    mic = MicrophoneStreamer()
    mic.start()
 
    rolling_buffer = np.zeros(0, dtype=np.float32)
 
    try:
        while True:
            chunk = mic.read_chunk()
            if chunk is None:
                continue
 
            rolling_buffer = np.concatenate([rolling_buffer, chunk])
 
            # Keep only the most recent BUFFER_SIZE samples (sliding window)
            if len(rolling_buffer) > BUFFER_SIZE:
                rolling_buffer = rolling_buffer[-BUFFER_SIZE:]
 
            # Wait until we have at least ~1 second of audio before running inference
            if len(rolling_buffer) < SAMPLE_RATE * 1.0:
                continue
 
            waveform = torch.tensor(rolling_buffer, device=device).unsqueeze(0)  # (1, T)
            length = torch.tensor([waveform.shape[1]], device=device)
 
            with torch.no_grad():
                # Preprocessor: raw audio -> log-mel spectrogram features
                processed_signal, processed_len = preprocessor(
                    input_signal=waveform, length=length
                )
 
                # Encoder: log-mel features -> frame-level embeddings
                encoded, encoded_len = encoder(
                    audio_signal=processed_signal, length=processed_len
                )
                # encoded shape: (batch, feature_dim, time) for NeMo -> transpose
                encoded = encoded.transpose(1, 2)  # (batch, time, feature_dim)
 
                # Classification head: frame embeddings -> per-frame language logits
                logits = lang_head(encoded)  # (batch, time, num_classes)
                preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
 
            # Print the predicted label sequence for this window
            # (~80ms per frame depending on model subsampling factor)
            label_seq = [LABELS[p] for p in preds]
            # Just show the last few frames to avoid flooding the terminal
            print(f"Last frames: {label_seq[-10:]}")
 
            time.sleep(0.05)
 
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
 
 
if __name__ == "__main__":
    main()
 
