import os
import re
import queue
import threading
import sounddevice as sd
from openai import OpenAI
from kokoro import KPipeline
from dotenv import load_dotenv

# 1. Load variables from the .env file
load_dotenv()

# 2. Access the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# 3. Initialize the OpenAI client using the key
client = OpenAI(api_key=openai_api_key)

class LLMStreamTTS:
    def __init__(self, lang_code='a', voice='af_heart', speed=1.0, sample_rate=24000):
        print("Initializing Kokoro TTS Engine...")
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()

    def _tts_producer(self):
        """
        Thread 1: Consumes complete text chunks from the text queue,
        generates Kokoro audio tensors, and pushes them to the audio queue.
        """
        while not self.stop_event.is_set():
            chunk = self.text_queue.get()
            if chunk is None:  # End-of-stream signal from LLM accumulator
                break

            # Filter out chunks without alphanumeric characters to prevent model noise
            if re.search(r'[a-zA-Z0-9]', chunk):
                generator = self.pipeline(chunk, voice=self.voice, speed=self.speed)
                for _, _, audio in generator:
                    if self.stop_event.is_set():
                        break
                    self.audio_queue.put(audio)
            
            self.text_queue.task_done()

        # Signal consumer that audio generation is done
        self.audio_queue.put(None)

    def _audio_consumer(self):
        """
        Thread 2: Holds the audio driver open continuously and streams audio data 
        directly to the speakers without hardware clicks or pops.
        """
        with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while not self.stop_event.is_set():
                audio = self.audio_queue.get()
                if audio is None:  # End signal
                    break

                # Reshape 1D audio array to 2D for sounddevice stream
                stream.write(audio.reshape(-1, 1))
                self.audio_queue.task_done()

    def process_llm_stream(self, prompt: str):
        """
        Streams response from LLM, accumulates tokens into complete clauses,
        and pushes them immediately into the TTS pipeline.
        """
        self.stop_event.clear()
        
        # Clear existing queues
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # Start TTS Producer and Audio Consumer threads
        producer = threading.Thread(target=self._tts_producer)
        consumer = threading.Thread(target=self._audio_consumer)
        producer.start()
        consumer.start()

        SYSTEM_PROMPT = """You are a voice assistant generating responses designed strictly for Text-to-Speech (TTS) synthesis. 

Follow these formatting rules without exception:

1. NO MARKDOWN OR FORMATTING:
   - Do NOT use headers (##, ###), bold text (**word**), italics (*word*), or code blocks (```).
   - Do NOT use bullet points (- or *) or numbered lists (1., 2.). Use natural verbal transitions like "First,", "Second,", "Additionally,", or "Finally," instead.
   - Do NOT output emojis, math symbols, or special ASCII characters.

2. EXPAND ALL ABBREVIATIONS AND SYMBOLS:
   - Write out "e.g." as "for example".
   - Write out "i.e." as "that is".
   - Write out "etc." as "et cetera" or "and so on".
   - Write out "vs." or "vs" as "versus".
   - Write out "&" as "and", "%" as "percent", and "$" as "dollars" (e.g., write "50 dollars", not "$50").
   - Write out URLs phonetically or simplify them (e.g., "example dot com").

3. NATURAL SPEECH PUNCTUATION:
   - Use standard punctuation (periods, commas, question marks, and exclamation points) to guide the reader's natural cadence and breath pauses.
   - Do NOT use em-dashes (—), hyphens for lists, or ellipses (...) as they can disrupt model chunking.

4. TONE AND STRUCTURE:
   - Speak in clear, conversational sentences.
   - Structure long answers into well-formed paragraphs with smooth narrative transitions."""

        # Request streaming completion from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        buffer = ""
        # Punctuation or break signals to trigger chunk synthesis
        split_pattern = re.compile(r'([.!?;\n]+)')

        print("\n[LLM Output & TTS Playback Started]\n")

        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)  # Print text to terminal in real time
            buffer += token

            # Check if buffer contains sentence boundaries or exceeds 90 characters
            while True:
                match = split_pattern.search(buffer)
                if match:
                    split_pos = match.end()
                    to_send = buffer[:split_pos].strip()
                    buffer = buffer[split_pos:]
                    
                    if to_send:
                        self.text_queue.put(to_send)
                elif len(buffer) > 90 and ' ' in buffer:
                    # If line is long without sentence marks, split at last space
                    last_space = buffer.rfind(' ')
                    to_send = buffer[:last_space].strip()
                    buffer = buffer[last_space:]
                    
                    if to_send:
                        self.text_queue.put(to_send)
                else:
                    break

        # Flush any remaining text left in the buffer when the LLM stream ends
        if buffer.strip():
            self.text_queue.put(buffer.strip())

        # Send termination signals down the pipeline
        self.text_queue.put(None)

        # Wait for speech playback to finish completely
        producer.join()
        consumer.join()
        print("\n\n[Finished Speaking]")


# --- Main Execution Loop ---
if __name__ == "__main__":
    llm_tts = LLMStreamTTS(voice='af_heart', speed=1.0)

    while True:
        try:
            user_prompt = input("\nEnter prompt for LLM: ")
            if user_prompt.strip().lower() in ['exit', 'quit']:
                break
            if not user_prompt.strip():
                continue

            llm_tts.process_llm_stream(user_prompt)

        except KeyboardInterrupt:
            print("\nStopping...")
            break