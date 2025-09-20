import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import assemblyai as aai
import ffmpeg
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from speechbrain.pretrained import EncoderClassifier

load_dotenv()

CONST_16000 = 16000

TOKEN = os.getenv("AAI_API_KEY")
VOICE_DB_FILE = "voice_db.json"
VOICE_DB_DIR = "./voice_db"  # dossier avec les fichiers audio de référence




aai.settings.api_key = TOKEN
config = aai.TranscriptionConfig(
    speaker_labels=True,
    language_code="fr",
    speakers_expected=5,
    content_safety=False,
    sentiment_analysis=False,
    speech_model="best"
)
aai.settings.http_timeout = 600



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TEST_MODE = os.getenv("DIARIZATION_TEST_MODE", "0") == "1"

def _get_classifier():
    if _TEST_MODE:
        class _Fake:
            def encode_batch(self, wav):
                import numpy as _np
                import torch as _torch
                return _torch.from_numpy(_np.ones((1,192), dtype=_np.float32))
        return _Fake()
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )


CLASSIFIER = _get_classifier()

def _load_wav_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)           # (channels, time)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono
    if sr != CONST_16000:
        wav = torchaudio.functional.resample(wav, sr, CONST_16000)
    return wav



def transform_mkv_to_mp3(input_file, output_file):
    try:
        print(f"[INFO] Conversion de {input_file} en {output_file}...")
        # Conversion de MKV en MP3 avec ffmpeg
        ffmpeg.input(input_file).output(output_file, format='mp3', acodec='libmp3lame').run(overwrite_output=True)
        print(f"[OK] Conversion réussie : {output_file}")
    except ffmpeg.Error as e:
        print(f"[ERROR] Erreur lors de la conversion : {e}")


def transcribe_with_retry(file_path, max_retries=3):
    print(f"[INFO] Tentative de transcription de {file_path}...")
    for attempt in range(max_retries):
        try:
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(
                file_path
            )
            print(f"[INFO] Transcription réussie pour {file_path}")
            return transcript
        except Exception as e:
            print(f"[ERROR] Erreur: {e}. Nouvelle tentative ({attempt+1}/{max_retries})...")
            time.sleep(5)
    raise Exception("Échec de la transcription après plusieurs tentatives.")


def transcribe_with_diarization(file_path: str) -> dict:
    transcript = transcribe_with_retry(file_path)
    # Textes dans l'ordres
    texts = []
    # Regroupement des segments par locuteur
    speakers = {}
    for utterance in transcript.utterances:
        if (speakers.get(utterance.speaker) is None):
            speakers[utterance.speaker] = []
        speakers[utterance.speaker].append({"text": utterance.text, "start": utterance.start, "end": utterance.end})
        texts.append({"speaker": utterance.speaker, "text": utterance.text})
    return speakers, texts

def write_transcript(file_path: str, speakers: dict, texts: list) -> None:
    with open(file_path, "w") as f:
        for text in texts:
            f.write(f"{speakers[text['speaker']]}: {text['text']}\n")

def calculate_longest_segment(speakers: dict) -> dict:
    print("[INFO] Calcul des segments les plus longs par locuteur...")
    longest_segments = {}
    for speaker, segments in speakers.items():
        longest_segment = max(segments, key=lambda x: x['end'] - x['start'], default=None)
        if longest_segment:
            longest_segments[speaker] = longest_segment
    print("[OK] Segments les plus longs calculés.")
    return longest_segments

#write_transcript("./transcript.txt", speakers, texts)
def extract_audio_sample(mp3_path: str, start_ms: int, end_ms: int, output_path: str):
    """Extrait un échantillon audio à partir de timestamps en millisecondes."""
    print(f"[INFO] Extraction de l'échantillon audio de {start_ms} à {end_ms} ms...")
    # Conversion en secondes avec précision milliseconde
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0

    subprocess.run([
        "ffmpeg", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i", mp3_path,
        "-acodec", "libmp3lame", output_path
    ], 
    check=True,        
    stdout=subprocess.DEVNULL,  # supprime sortie standard
    stderr=subprocess.DEVNULL   # supprime sortie erreur
    )
    print(f"[OK] Échantillon extrait : {output_path}")

# === FONCTIONS POUR BASE DE DONNÉES VOCALE ===

def _sim(a: np.ndarray, b: np.ndarray) -> float:
    # similarité cosinus dans [ -1, 1 ] (≈ 0.0–1.0 en pratique)
    return 1.0 - float(cosine(a, b))

def get_embedding(path: str) -> np.ndarray:
    wav = _load_wav_mono_16k(path)                  # (1, time)
    emb = CLASSIFIER.encode_batch(wav)              # (batch, 192) ou (batch, 1, 192)
    emb = emb.squeeze()                             # enlève les dims de taille 1
    emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32).ravel() # -> (192,)
    wav = _load_wav_mono_16k(path)                  # (1, time)
    return _get_classifier().encode_batch(wav)              # (batch, 192) ou (batch, 1, 192)

def build_similarity_matrix(sample_paths: dict[str, str],
                            db: dict[str, np.ndarray],
                            candidate_names: list[str] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """Construit une matrice SxK avec S=nb speakers détectés, K=nb joueurs candidats.
       sample_paths: {"A": "tmp_A.mp3", "B": "tmp_B.mp3", ...}
       candidate_names: restreint la base aux 5 joueurs si la DB a plus de monde.
    """
    speakers = list(sample_paths.keys())
    names = candidate_names if candidate_names else list(db.keys())

    # embeddings des samples
    sample_embs = []
    for spk in speakers:
        e = get_embedding(sample_paths[spk])           # (192,)
        e = np.asarray(e, dtype=np.float32).ravel()
        sample_embs.append(e)

    # embeddings des joueurs
    name_embs = []
    for name in names:
        e = np.asarray(db[name], dtype=np.float32).ravel()
        name_embs.append(e)

    # matrice de similarité SxK
    S, K = len(speakers), len(names)
    sim = np.zeros((S, K), dtype=np.float32)
    for i in range(S):
        for j in range(K):
            sim[i, j] = _sim(sample_embs[i], name_embs[j])

    return sim, speakers, names

def assign_speakers_hungarian(sim: np.ndarray, speakers: list[str], names: list[str],
                              min_similarity: float = 0.45) -> dict[str, tuple[str | None, float]]:
    """Retourne un mapping speaker -> (nom_assigné_ou_None, similarité).
       Utilise Hungarian sur le coût = 1 - similarité. Impose 1-1.
    """
    # coût à minimiser
    cost = 1.0 - sim

    # Hungarian
    row_ind, col_ind = linear_sum_assignment(cost)  # indices des paires optimales

    mapping: dict[str, tuple[str | None, float]] = {}
    for r, c in zip(row_ind, col_ind, strict=False):
        spk = speakers[r]
        name = names[c]
        s = float(sim[r, c])
        if s >= min_similarity:
            mapping[spk] = (name, s)
        else:
            mapping[spk] = (None, s)  # pas assez fiable → à traiter comme "inconnu"
    return mapping

def _name_from_path(p: Path) -> str:
    if p.parent.name != "voice_db":      # cas dossiers par personne
        return p.parent.name
    # cas plat: "Alice_..." -> "Alice"
    m = re.match(r"([^-_]+)[-_].*", p.stem)
    return m.group(1) if m else p.stem   # fallback: nom=stem complet

def build_voice_db(db_dir: str, out_json="voice_db.json"):
    db_dir = Path(db_dir)
    per_name: dict[str, list[np.ndarray]] = {}
    audio_ext = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

    files = [p for p in db_dir.rglob("*") if p.suffix.lower() in audio_ext]
    print(f"[INFO] Fichiers détectés: {len(files)}")

    for i, p in enumerate(files, 1):
        name = _name_from_path(p)
        emb = get_embedding(str(p))  # <- (192,)
        per_name.setdefault(name, []).append(emb)
        print(f"[{i}/{len(files)}] {name} <= {p.name}")

    # moyenne par personne
    voice_db = {
        name: np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32).tolist()
        for name, vecs in per_name.items()
    }

    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(voice_db, w)
    print(f"[OK] Base vocale écrite: {out_json} ({len(voice_db)} personnes)")

def load_voice_db(db_json: str = "voice_db.json") -> dict:
    with open(db_json, encoding="utf-8") as r:
        db = json.load(r)
    # force 1D
    for k, v in db.items():
        db[k] = np.asarray(v, dtype=np.float32).ravel()
    return db

def match_voice(sample_path: str, db_json: str = "voice_db.json",
                min_similarity: float = 0.45) -> tuple[str | None, float]:
    db = load_voice_db(db_json)
    sample = get_embedding(sample_path)   # (192,)
    best_name, best_sim = None, -1.0
    for name, ref in db.items():
        # scipy.cosine -> distance ; similarité ~ 1 - distance
        sim = 1.0 - float(cosine(sample, ref))
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < min_similarity:
        return None, best_sim
    return best_name, best_sim


def get_episode_name(file: str) -> str:
    """ Renvoie le nom de l'épisode"""
    return Path(file).stem



if __name__ == "__main__":
    # --- (optionnel) liste des 5 joueurs attendus pour restreindre la DB ---
    KNOWN_PLAYERS = None
    MIN_SIMILARITY = 0.45  

    # 0) Build DB si absente
    if not os.path.exists(VOICE_DB_FILE):
        build_voice_db(VOICE_DB_DIR, out_json=VOICE_DB_FILE)

    # 1) Entrées
    episode_number = 24
    file_mp3 = f"../datas/Episode {episode_number}.mp3"
    file_mkv = f"../datas/Episode {episode_number}.mkv"
    episode_name = get_episode_name(file_mp3)

    # 2) Conversion MKV -> MP3
    transform_mkv_to_mp3(file_mkv, file_mp3)
    # 3) Transcription + diarization
    speakers, texts = transcribe_with_diarization(file_mp3)

    # 4) Plus long segment par speaker
    longest_segment = calculate_longest_segment(speakers)

    # 5) Extraction des échantillons pour chaque speaker
    os.makedirs("./to_class", exist_ok=True)
    samples = {}  # {speaker_id: path_sample}
    for speaker, seg in longest_segment.items():
        tmp_path = f"./to_class/tmp_{speaker}.mp3"
        extract_audio_sample(file_mp3, seg["start"], seg["end"], tmp_path)
        samples[speaker] = tmp_path

    # 6) Chargement de la base de voix
    db = load_voice_db(VOICE_DB_FILE)  # {name: embedding(1D)}
    if KNOWN_PLAYERS is not None:
        # restreint la DB aux 5 joueurs (si elle contient d'autres personnes)
        db = {k: v for k, v in db.items() if k in set(KNOWN_PLAYERS)}
        if len(db) == 0:
            raise RuntimeError("KNOWN_PLAYERS ne correspond à aucune entrée dans la base vocale.")

    # 7) Matrice de similarité et assignation hongroise (1–à–1)
    sim, speakers_order, candidate_names = build_similarity_matrix(samples, db, candidate_names=KNOWN_PLAYERS)
    mapping = assign_speakers_hungarian(sim, speakers_order, candidate_names, min_similarity=MIN_SIMILARITY)
    # mapping: { "A": ("NomJoueur" | None, score) }

    print("\n[ASSIGNATIONS (Hongrois)]")
    for spk in speakers_order:
        name, s = mapping.get(spk, (None, 0.0))
        print(f"  {spk:>8} -> {name or 'INCONNU'}  (sim={s:.3f})")

    # 8) Renommage des fichiers et mapping final pour l'écriture
    matched_speakers = {}  # {speaker_id: nom_affiché}
    for spk, (name, _) in mapping.items():
        if name is not None:
            new_name = f"./to_class/{name}_{episode_name}.mp3"
            try:
                os.replace(samples[spk], new_name)
            except Exception:
                # si le fichier existe déjà, ajoute un suffixe
                base, ext = os.path.splitext(new_name)
                k = 2
                while os.path.exists(f"{base}_{k}{ext}"):
                    k += 1
                os.replace(samples[spk], f"{base}_{k}{ext}")
            matched_speakers[spk] = name
        else:
            # conserve le label d'origine mais marque 'inconnu'
            matched_speakers[spk] = f"{spk} (inconnu)"

    # 9) Remplacement des labels dans la transcription + sauvegarde
    write_transcript(f"./{episode_name}.txt", matched_speakers, texts)
    print(f"[OK] Transcription annotée écrite dans ./{episode_name}.txt")
    sys.exit()

