install-w-venv:
	uv run --with qg -- python -c "import qg"

install:
	uv pip install -e .

test:
	python3 -m pytest src/qg/test.py

# Generate datasets with Hydra + WandB tracking
generate: install
	python -m qg.train

# Generate specific scenario
generate-decaying: install
	python -m qg.train scenario=decaying_turbulence

generate-fpc: install
	python -m qg.train scenario=flow_past_cylinder

generate-fpc-sponge: install
	python -m qg.train scenario=flow_past_cylinder_sponge

generate-cape: install
	python -m qg.train scenario=cape_high_re

generate-forced: install
	python -m qg.train scenario=forced_turbulence

# Quick test (small grid, offline)
test-generate: install
	python -m qg.train \
		scenario=decaying_turbulence \
		grid.Nx=64 \
		grid.Ny=64 \
		time.T=10 \
		ic.n_batch=2 \
		wandb.mode=offline

# Clean output
clean:
	rm -rf runs/