import collections
import queue
import sys
import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import sounddevice as sd

# 1. Configuration Constants
MODEL_ID = "facebook/mms-lid-126"
SAMPLE_RATE = 16000      
WINDOW_DURATION = 4.0    
STEP_DURATION = 1.0      

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_DURATION)

audio_queue = queue.Queue()

# 2. Define our target languages (English + Major Indian Languages)
# ISO 639-3 standard codes used by the MMS model
TARGET_LANGUAGES = {
    "eng", # English
    "hin", # Hindi
    "ben", # Bengali
    "tel", # Telugu
    "mar", # Marathi
    "tam", # Tamil
    "urd", # Urdu
    "guj", # Gujarati
    "kan", # Kannada
    "mal", # Malayalam
    "ory", # Odia
    "pan", # Punjabi
    "asm", # Assamese
}

print("Loading Meta MMS model...")
processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Pre-compute the allowed indices (Fixing the NoneType error)
# The MMS config lacks label2id, so we reverse-engineer it from id2label
label2id = {lang_code: int(lang_id) for lang_id, lang_code in model.config.id2label.items()}

allowed_indices = []
for lang in TARGET_LANGUAGES:
    if lang in label2id:
        allowed_indices.append(label2id[lang])
    else:
        print(f"Warning: {lang} not found in model vocabulary.")

# Move allowed indices to the same device as the model
allowed_indices_tensor = torch.tensor(allowed_indices, device=device)

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def main():
    rolling_buffer = collections.deque(maxlen=WINDOW_SAMPLES)
    rolling_buffer.extend(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    
    print("\n=== Initializing Microphone Stream ===")
    print(f"Restricted to detecting: {', '.join(TARGET_LANGUAGES)}")
    print("Speak into your microphone. Press Ctrl+C to terminate.")
    
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,            
        callback=audio_callback,
        blocksize=STEP_SAMPLES 
    )
    
    with stream:
        while True:
            try:
                data_chunk = audio_queue.get()
                rolling_buffer.extend(data_chunk[:, 0])
                
                audio_window = np.array(rolling_buffer, dtype=np.float32)
                
                inputs = processor(audio_window, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # --- LOGIT MASKING APPLIED HERE ---
                logits = outputs.logits
                
                # Create a mask filled with negative infinity
                masked_logits = torch.full_like(logits, float('-inf'))
                
                # Copy over the true logit values ONLY for our target languages
                masked_logits[0, allowed_indices_tensor] = logits[0, allowed_indices_tensor]
                
                # Perform argmax on the masked logits, ensuring the result is always a target language
                predicted_id = torch.argmax(masked_logits, dim=-1).item()
                predicted_iso = model.config.id2label[predicted_id]
                
                sys.stdout.write(f"\rDetected Language (Filtered): \033[1;32m{predicted_iso}\033[0m ")
                sys.stdout.flush()
                
            except KeyboardInterrupt:
                print("\nStream stopped by user.")
                break
            except Exception as e:
                print(f"\nError encountered during inference: {e}")
                break

if __name__ == "__main__":
    main()