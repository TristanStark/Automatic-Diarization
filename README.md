# Automatic Diarization

Pipeline de diarization + matching de voix pour identifier automatiquement les intervenants d'un enregistrement (FR).

## Fonctionnalités
- Transcodage MKV -> MP3 (ffmpeg)
- Transcription + diarization (AssemblyAI)
- Extraction d'échantillons par locuteur
- Empreintes vocales (speechbrain ECAPA) + matching cosinus
- Assignation **1–à–1** via l’algorithme hongrois
- Construction automatique de la base vocale si absente

## Prérequis
- Python 3.10+
- FFmpeg installé
- Clé API AssemblyAI

## Installation
```bash
cp .env.sample .env   # renseigner AAI_API_KEY
pip install -e .
