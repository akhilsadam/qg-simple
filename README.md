## Quickstart
- First, clone or fork the `package-stable` branch.
- Create an environment (with `uv` from `pip install uv`).
- Then install with `make install`
- Run an example with `make test-generate`

You may need an appropriate version of FFMPEG, and sufficiently up-to-date Linux.
Note example config files are in `conf/` and can be edited as presets.

If `CUDA` is not automatically detected, you may need to set a preset variable in `conf/config.yaml`

## Usage
This code is provided as a library-style package, so you should import the `qg` package after the quickstart to use it in non-preset ways.
See `__init__.py` for a simple example with `direct_solver`, that exposes the `QG` instance with functions in `qg`.


## Changelog [`package-variant`]:

### 0.2.1 (2026-02-18)
- Refactored derivative, grid, and input code
- Made space for Zeitlin-style sine derivatives

### 0.2.0 (2026-02-16)
- Update to work with Hydra and updated Mura
- Corrected streamfunction convention
- Started merging other QG codes
- Still unpublished to pypi

### 0.1.0 (2025-07-14)
- Initial package version
- Refactored to work with simpler configuration (new mura package @ 0.2.7 that uses dataclasses)
