PY ?= .venv/bin/python

.PHONY: setup l1 l2 l3 l4 l5 l6 l7 l8 all serve test lint data gate

setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

l1 l2 l3 l4 l5 l6 l7 l8 all:
	$(PY) -m jevpoc.run $@

serve:
	$(PY) -m uvicorn jevpoc.gateway.app:app --port 8000 --reload

data:
	$(PY) -m jevpoc.eval.data

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

gate:
	$(PY) -m jevpoc.eval.protocol && $(PY) -m jevpoc.eval.gate
