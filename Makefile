install-w-venv:
	uv run --with qg -- python -c "import qg"

install:
	uv pip install -e .

test:
	python3 -m pytest src/qg/test.py