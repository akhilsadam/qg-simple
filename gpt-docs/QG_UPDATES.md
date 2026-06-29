# QG Package Updates from qg-2d and obs-closure

## Summary of Changes

Updated the QG package with improvements from qg-2d and qg-2d-obs-closure repositories while maintaining the clean Hydra-based structure.

## New Features

### 1. Sponge Layer Support
- **Purpose**: Boundary damping for open boundary conditions
- **Config**: `SpongeConfig` dataclass with x/y boundary positions
- **Parameters**: 
  - `sponge_coeff`: Damping coefficient (typically scaled by dt)
  - Boundaries: `x_left`, `x_right`, `y_top`, `y_bottom` (normalized 0-1)
- **Example**: See `flow_past_cylinder_sponge.yaml` scenario

### 2. Turbulence Closure Models
- **Options**: 
  - `closure_option=1`: Smagorinsky model
  - `closure_option=2`: Leith model
  - `closure_option=None`: No closure (default)
- **Parameter**: `closure_c` - closure coefficient (default 0.1)
- **Example**: See `forced_turbulence.yaml` scenario

### 3. FPC (Flow Past Cylinder) Initial Conditions
- **Purpose**: Inflow boundary conditions for obstacle flows
- **Config**: `ic.option=2` with `init_vel` parameter
- **Parameters**:
  - `init_vel`: Initial inflow velocity (can be negative/positive)
- **Example**: See `flow_past_cylinder_sponge.yaml`

### 4. Enhanced Mask Configuration
- **New parameters**:
  - `x_center`, `y_center`: Normalized obstacle position (0-1)
  - `option`: Mask type (1=circular, 2=image, 3=netCDF)
  - `mask_file`: Path for image/netCDF masks
  - `blur`: Gaussian blur for image masks
- **Updated defaults**: `r = π/5` (more realistic obstacle size)

## Parameter Updates

### Renamed Parameters
- `save_rate` → `save_int` (time stepping)
- `penalty` → `penalty_coeff` (PDE parameters)
- Both old and new names supported for backward compatibility

### Updated Defaults (from qg-2d)
- `nu`: 1.025e-5 → 1.025e-4 (more realistic viscosity)
- `seed`: 42 → 86 (matches qg-2d default)
- `ic.seed`: Added separate seed for initial conditions
- `mask.r`: π/4 → π/5 (smaller obstacle)

### New PDE Parameters
- `sponge_coeff`: Sponge layer damping coefficient
- `closure_option`: Turbulence closure model selection
- `closure_c`: Closure model coefficient

## Configuration Examples

### Flow Past Cylinder with Sponge
```yaml
qg:
  pde:
    nu: 5e-3  # RE 200
    penalty_coeff: 1.25e-3  # 1.25 * dt
    sponge_coeff: 0.125  # 125 * dt
  
  ic:
    option: 2  # FPC
    init_vel: -2
  
  mask:
    option: 1
    r: 0.628  # π/5
    x_center: 0.25
    y_center: 0.5
  
  sponge:
    option: 1
    x_left: 0.025
    x_right: 0.975
```

### Forced Turbulence with Closure
```yaml
qg:
  pde:
    mu: 0.02
    nu: 1.025e-4
    B: 2.5
    closure_option: 2  # Leith
    closure_c: 0.1
  
  forcing:
    A: -0.1
    B: 2
    D: 0.1
    E: 2
```

## Backward Compatibility

All changes maintain backward compatibility:
- Old parameter names still work (`save_rate`, `penalty`)
- Legacy `function` parameter coexists with new `option`
- Hydra config conversion handles both old and new formats

## New Scenario Configs

1. **flow_past_cylinder_sponge.yaml**: FPC with sponge boundaries (from obs-closure)
2. **forced_turbulence.yaml**: Updated with closure model and qg-2d defaults
3. **decaying_turbulence.yaml**: Updated with qg-2d seed default

## Testing

Use Hydra sweeps to test new features:
```bash
# Test sponge layers
python -m qg.test scenario=flow_past_cylinder_sponge

# Test closure models
python -m qg.test scenario=forced_turbulence

# Sweep over closure options
python -m qg.test --multirun scenario=forced_turbulence qg.pde.closure_option=null,1,2
```
