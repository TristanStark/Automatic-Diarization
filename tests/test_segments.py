import os
import json
import numpy as np
from pathlib import Path
import importlib

def test_build_voice_db_and_similarity(tmp_path, monkeypatch):
    diar = importlib.import_module("diarization")

    db_json = os.getenv("VOICE_DB_FILE", "voice_db.json")
    if not Path(db_json).exists():
        diar.build_voice_db(os.getenv("VOICE_DB_DIR", "./voice_db"), out_json=db_json)

    assert Path(db_json).exists()
    with open(db_json, "r", encoding="utf-8") as r:
        db = json.load(r)
    assert set(db.keys()) >= {"Alice", "Bob"}
    # vecteurs (192,)
    assert len(np.asarray(db["Alice"]).ravel()) == 192

    # Matrice de similarité sur des samples factices
    samples = {"S1": "voice_db/Alice_1.wav", "S2": "voice_db/Bob_1.wav"}
    sim, speakers, names = diar.build_similarity_matrix(samples, diar.load_voice_db(db_json))
    assert sim.shape == (2, len(names))
    assert set(speakers) == {"S1", "S2"}

def test_calculate_longest_segment_simple():
    from diarization import calculate_longest_segment
    speakers = {
        "A": [{"start": 0, "end": 1000}, {"start": 1000, "end": 2500}],
        "B": [{"start": 0, "end": 500}],
    }
    longest = calculate_longest_segment(speakers)
    assert longest["A"]["end"] - longest["A"]["start"] == 1500
    assert longest["B"]["end"] - longest["B"]["start"] == 500
