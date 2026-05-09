export PYTHONPATH = .
export LOG_FILE = logs/main.log

markup:
	LOG_FILE=logs/markup.log python entry/markup.py

train:
	LOG_FILE=logs/train.log python entry/train.py

train-test:
	LOG_FILE=logs/train-test.log python entry/train.py --test-run

evaluate:
	LOG_FILE=logs/evaluate.log python entry/evaluate_models.py

test:
	python -m unittest discover tests

build-dataset:
	LOG_FILE=logs/build-dataset.log python scripts/build_dataset.py

build-pos-accent-db:
	LOG_FILE=logs/build-pos-accent-db.log python scripts/build_pos_accent_db.py

build-pos-accent-db-test:
	LOG_FILE=logs/build-pos-accent-db-test.log python scripts/build_pos_accent_db.py --test-run

build-grammar-db:
	LOG_FILE=logs/build-grammar-db.log python scripts/build_grammar_db.py

build-grammar-db-test:
	LOG_FILE=logs/build-grammar-db-test.log python scripts/build_grammar_db.py --test-run

evaluate-accentuator:
	LOG_FILE=logs/evaluate-accentuator.log python scripts/evaluate_accentuator.py
