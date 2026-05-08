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

