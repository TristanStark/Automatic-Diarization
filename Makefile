PY=python
PIP=pip

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  venv          - create venv"
	@echo "  install       - install deps"
	@echo "  lint          - run ruff"
	@echo "  test          - run pytest (mode test)"
	@echo "  run           - run diarization.py"
	@echo "  db            - (re)build voice DB"
	@echo "  docker-build  - build docker image"
	@echo "  docker-run    - run container"
	@echo "  clean         - remove caches"

venv:
	$(PY) -m venv .venv

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

lint:
	ruff check .

test:
	DIARIZATION_TEST_MODE=1 pytest

run:
	$(PY) diarization.py

db:
	$(PY) -c "import os; os.environ.setdefault('DIARIZATION_TEST_MODE', '1'); \
import diarization as d; d.build_voice_db(os.getenv('VOICE_DB_DIR','./voice_db'), out_json=os.getenv('VOICE_DB_FILE','voice_db.json'))"

docker-build:
	docker build -t automatic-diarization:latest .

docker-run:
	docker run --rm --env-file .env -v $(PWD)/datas:/app/datas -v $(PWD)/voice_db:/app/voice_db automatic-diarization:latest

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ build dist *.egg-info
