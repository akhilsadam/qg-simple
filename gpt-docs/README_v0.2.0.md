# QG (Quasi-Geostrophic Flow) v0.2.0

Physics-based turbulence simulation with Hydra configuration and WandB experiment tracking.

## What's New in v0.2.0

✨ **Hydra Configuration System** - Compositional configs for different scenarios  
📊 **WandB Integration** - Full experiment tracking with artifact management  
🔧 **Mura Utilities** - Git tracking, version management  
🎯 **Scenario-Based** - Pre-configured physics scenarios  
✅ **Backward Compatible** - Old `test.py` still works

---

## Installation

```bash
cd qg
make install

# Or with uv
uv pip install -e .
```

**Dependencies:** Automatically installs Mura v0.3.0 with Hydra utilities.

---

## Quick Start

### Generate Datasets with Tracking

```bash
# Decaying turbulence (default)
make generate

# Or directly
python -m qg.train

# Specific scenarios
make generate-decaying    # Decaying turbulence
make generate-fpc         # Flow past cylinder
make generate-cape        # Cape flow (high Re)
make generate-forced      # Forced turbulence

# Quick test (small, offline)
make test-generate
```

### Customize Parameters

```bash
# Override any parameter via CLI
python -m qg.train \
    scenario=decaying_turbulence \
    grid.Nx=512 \
    grid.Ny=512 \
    time.T=100 \
    pde.nu=1e-4

# Different scenario with custom params
python -m qg.train \
    scenario=flow_past_cylinder \
    grid.Nx=512 \
    pde.nu=0.01 \
    time.T=200
```

---

## Available Scenarios

### 1. Decaying Turbulence (default)
**Use:** ML dataset generation, benchmark testing

```yaml
scenario: decaying_turbulence
grid: 256x256
Re: ~1000 (nu=1.025e-5)
Features: No forcing, periodic BC
```

### 2. Flow Past Cylinder
**Use:** Obstacle flow, wake dynamics

```yaml
scenario: flow_past_cylinder
grid: 256x256
Re: 200
Features: Circular obstacle, outflow BC
```

### 3. Cape Flow (High Re)
**Use:** Complex geometry, high Re flows

```yaml
scenario: cape_high_re
grid: 1024x512
Re: 2000
Features: Cape obstacle, sponge BC
```

### 4. Forced Turbulence
**Use:** Statistically steady turbulence

```yaml
scenario: forced_turbulence
grid: 256x256
Features: Energy injection, long time
```

---

## Configuration System

### Hydra Composition

Create custom scenarios in `src/qg/conf/scenario/`:

```yaml
# my_scenario.yaml
name: my_scenario

grid:
  Nx: 128
  Ny: 128

time:
  dt: 0.001
  T: 50
  save_rate: 100

pde:
  nu: 1e-5
  
ic:
  n_batch: 10
```

Use it:
```bash
python -m qg.train scenario=my_scenario
```

### Override Hierarchy

```bash
# CLI overrides scenario
python -m qg.train scenario=decaying_turbulence grid.Nx=512

# Multiple overrides
python -m qg.train \
    scenario=flow_past_cylinder \
    grid.Nx=512 \
    pde.nu=0.001 \
    time.T=200
```

---

## WandB Integration

### Automatic Tracking

Every run logs:
- Full configuration
- Git SHA and dirty status
- Dataset statistics (shape, mean, std, min, max)
- Generated videos
- Output artifacts

### Configuration

```bash
# Online mode (default)
python -m qg.train wandb.mode=online

# Offline mode (sync later)
python -m qg.train wandb.mode=offline

# Disabled (no WandB)
python -m qg.train wandb.mode=disabled

# Custom project
python -m qg.train wandb.project=my-qg-project
```

### Artifacts

Each run uploads:
- `qg_data.npy` - Full tensor dataset
- `config.yaml` - Complete configuration
- `*.mp4` - Visualization videos
- `git_diff.patch` - If repo is dirty

---

## Output Structure

```
runs/
└── qg/
    └── decaying_turbulence/
        └── A1B2C3_add-hydra-config/
            ├── .hydra/
            │   └── config.yaml       # Full resolved config
            ├── qg_data.npy           # Dataset [B, T, C, H, W]
            ├── qg_data.mp4           # Visualization
            ├── qg_data_clamped.mp4
            ├── qg_data_seismic.mp4
            ├── config.yaml           # Saved config
            └── git_diff.patch        # If dirty
```

---

## Dataset Format

Generated `.npy` file has shape `[B, T, C, H, W]`:
- **B**: Batch size (`ic.n_batch`)
- **T**: Number of saved timesteps (`time.T / time.dt / time.save_rate`)
- **C**: 4 channels (vorticity, u-velocity, v-velocity, streamfunction)
- **H**: Grid height (`grid.Ny`)
- **W**: Grid width (`grid.Nx`)

Load with:
```python
import numpy as np
data = np.load('runs/qg/.../qg_data.npy')
print(data.shape)  # e.g., [20, 600, 4, 256, 256]
```

---

## Integration with Autoencoders

QG is automatically used by autoencoders for ML training:

```bash
# In autoencoders directory
python -m src.autoencoders.train data=qg_turbulence model=tiny_cu
```

The autoencoder datamodule:
1. Checks cache for matching params
2. If not found, generates via QG solver
3. Caches for future use
4. Returns PyTorch dataloaders

---

## Legacy API (Still Works)

Old test functions still work for backward compatibility:

```python
from qg import config, autorun

cfg = config()
cfg.logging.task_name = "my_test"
# ... configure ...
autorun(cfg)
```

But new Hydra-based approach is recommended for better tracking!

---

## Physics Parameters

### Grid
- `Nx`, `Ny`: Resolution (powers of 2 recommended)
- `Lx`, `Ly`: Domain size (default 2π)

### Time Integration
- `dt`: Timestep (stability: dt < 1/(nu*k^2) where k is max wavenumber)
- `T`: Total simulation time
- `save_rate`: Save every N timesteps

### PDE Parameters
- `nu`: Kinematic viscosity (controls Reynolds number)
- `mu`: Linear drag coefficient
- `B`: Beta-plane parameter (Coriolis)
- `penalty`: Brinkman penalty for obstacles

### Initial Conditions
- `function`: 'randn' (random) or 'prescribed'
- `energy`: Initial kinetic energy
- `wavenumbers`: [k_min, k_max] for energy injection
- `n_batch`: Number of independent samples

---

## Examples

### High-Resolution Dataset
```bash
python -m qg.train \
    scenario=decaying_turbulence \
    grid.Nx=1024 \
    grid.Ny=1024 \
    ic.n_batch=50 \
    time.T=100
```

### Parameter Sweep
```bash
# Sweep over viscosity
python -m qg.train -m \
    scenario=decaying_turbulence \
    pde.nu=1e-5,5e-5,1e-4
```

### Debug Run (Fast)
```bash
python -m qg.train \
    scenario=decaying_turbulence \
    grid.Nx=64 \
    grid.Ny=64 \
    time.T=5 \
    ic.n_batch=2 \
    wandb.mode=disabled
```

---

## Troubleshooting

### "Mura not found"
```bash
cd ../mura && uv pip install -e .
```

### Simulation diverges
- Reduce `dt`
- Increase `nu` (viscosity)
- Check CFL condition

### Out of memory
- Reduce `grid.Nx`, `grid.Ny`
- Reduce `ic.n_batch`
- Reduce `time.T` or increase `save_rate`

---

## Development

### Run Tests
```bash
make test
```

### Add New Scenario
1. Create `src/qg/conf/scenario/my_scenario.yaml`
2. Run: `python -m qg.train scenario=my_scenario`

### Custom Physics
Override in config or CLI:
```bash
python -m qg.train \
    pde.nu=1e-4 \
    pde.mu=0.1 \
    pde.B=1.0
```

---

## Credits

**Author:** Akhil Sadam  
**Dependencies:** PyTorch, Hydra, Mura, WandB, jpcm

---

## Migration from v0.1.0

**Backward compatible!** Old test functions still work.

**New recommended workflow:**
```bash
# Old way (still works)
python -m pytest src/qg/test.py::test_decaying_qg_turbulence

# New way (better tracking)
python -m qg.train scenario=decaying_turbulence
```

See autoencoders `MIGRATION_GUIDE.md` for integration details.
