install-w-venv:
	uv run --with qg -- python -c "import qg"

install:
	uv pip install -e .

test:
	python3 -m pytest src/qg/test.py

# Quick test (small grid, offline)
test-generate: install
	python -m qg.train \
		scenario=forced_turbulence \
		qg.grid.Nx=128 \
		qg.grid.Ny=128 \
		qg.ic.n_batch=1 \
		wandb.mode=offline

test-generate-free: install
	python -m qg.train \
		scenario=smooth_cylinder \
		qg.ic.n_batch=1 \
		wandb.mode=offline

test-generate-dipole: install
	python -m qg.train \
		scenario=dipole \
		qg.ic.n_batch=1 \
		wandb.mode=offline

test-generate-box: install
	python -m qg.train \
		scenario=box \
		qg.ic.n_batch=1 \
		wandb.mode=offline

test-generate-split: install
	python -m qg.train \
		scenario=flow_past_cylinder \
		qg.grid.Nx=128 \
		qg.grid.Ny=128 \
		qg.ic.n_batch=1 \
		qg.integrator.split_bc=True \
		wandb.mode=offline

# Clean output
clean:
	rm -rf runs/