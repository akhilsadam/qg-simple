# Autodifferentiable QG code
- an evolving version of the solver code for our IEEE Oceans & JAMES papers
- see a [simple, quickstart version instead](https://github.com/akhilsadam/qg-simple)
- see the (limited) [official archival version](https://zenodo.org/records/17282193) or a [GitHub mirror instead](https://github.com/ananthu545/qg-2d)

## Quickstart
- First, clone or fork the `package-stable` branch.
- Create an environment (with `uv` from `pip install uv`).
- Then install with `make install`
- Run an example with `make test-generate`

You may need an appropriate version of FFMPEG, and sufficiently up-to-date Linux.

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
