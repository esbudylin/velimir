export PYTHONPATH = .

markup:
	python entry/markup.py

train:
	python entry/train.py

train-test:
	python entry/train.py --test-run

evaluate:
	python entry/evaluate_models.py

test:
	python -m unittest discover tests

build-dataset:
	python scripts/build_dataset.py

build-pos-accent-db:
	python scripts/build_pos_accent_db.py

build-pos-accent-db-test:
	python scripts/build_pos_accent_db.py --test-run

evaluate-accentuator:
	python scripts/evaluate_accentuator.py

