install-w-venv:
	uv run --with qg -- python -c "import qg"

install:
	uv pip install -e .

test:
	python3 -m pytest src/qg/test.py

# Quick test (small grid, offline)
test-generate: install
	python -m qg.train \
		scenario=decaying_turbulence \
		qg.grid.Nx=128 \
		qg.grid.Ny=128 \
		qg.ic.n_batch=1 \
		wandb.mode=offline

# Clean output
clean:
	rm -rf runs/