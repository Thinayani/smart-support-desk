.PHONY: install run test data audit train eval docker

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	python -m pytest -q

data:
	python ml/download_data.py

audit:
	python ml/audit.py

train:
	python ml/train.py

eval:
	python ml/eval_replies.py

docker:
	docker compose up --build
