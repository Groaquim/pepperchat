import io
import os
import wave
import threading
import numpy as np
from openai import OpenAI
from oai_dialogue.speech_to_text.transcriber import Transcriber, Transcript, TranscriptWord

class GroqTranscriber(Transcriber):
    def __init__(self, language="fr"):
        super().__init__()
        self.language = language
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self._buffer = []
        self._lock = threading.Lock()

    def push_pcm16_frames(self, sample_rate: int, channel_cnt: int, frames: np.ndarray):
        with self._lock:
            self._buffer.append((sample_rate, frames))

    def flush_and_transcribe(self, start_time: float, duration: float):
        with self._lock:
            if not self._buffer:
                return
            sample_rate = self._buffer[0][0]
            all_frames = np.concatenate([f for _, f in self._buffer])
            self._buffer.clear()

        def _worker():
            try:
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(all_frames.astype(np.int16).tobytes())
                wav_io.seek(0)
                wav_io.name = "audio.wav"

                res = self.client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=wav_io,
                    language=self.language
                )
                text = res.text.strip()
                if text:
                    self._on_transcribed(
                        start_time=start_time,
                        duration=duration,
                        transcripts=[Transcript(text, 1.0, [TranscriptWord(text, 1.0)])],
                        is_final=True,
                        additional_data=res
                    )
            except Exception as e:
                print("Groq Whisper error:", e)

        threading.Thread(target=_worker, daemon=True).start()