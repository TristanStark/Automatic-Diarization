import os
import wave
import contextlib
from pathlib import Path

def _write_silence_wav(path: Path, seconds=0.2, sr=16000):
    import numpy as np
    n_frames = int(seconds * sr)
    data = (np.zeros(n_frames)).astype("int16").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data)

def pytest_sessionstart(session):
    os.environ["DIARIZATION_TEST_MODE"] = "1"
    os.environ.setdefault("VOICE_DB_DIR", "./voice_db")
    os.environ.setdefault("VOICE_DB_FILE", "voice_db.json")
    Path("voice_db").mkdir(exist_ok=True)
    # Deux voix factices
    _write_silence_wav(Path("voice_db/Alice_1.wav"))
    _write_silence_wav(Path("voice_db/Bob_1.wav"))
