cd ..
conda create --prefix .sw python=3.10.15 git tmux graphviz conda-forge::ffmpeg
conda activate .sw
python3 -m pip install uv
uv venv
source .venv/bin/activate && uv pip install -r src/install/requirements.txt

# don't forget to set paths!
# USER_CONDA_ENV_PATH=/projects/ml_scope/Students/a1744874/scale_invariants/sw
# USER_VENV_PATH=/projects/ml_scope/Students/a1744874/scale_invariants/.venv