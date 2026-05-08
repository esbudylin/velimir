export PYTHONPATH = .

markup:
	python entry/markup.py

train:
	python entry/train.py

evaluate:
	python entry/evaluate_models.py

test:
	python -m unittest discover tests

test-train:
	python entry/train.py --test-run

build-dataset:
	python scripts/build_dataset.py

build-pos-accent-db:
	python scripts/build_pos_accent_db.py

evaluate-accentuator:
	python scripts/evaluate_accentuator.py

