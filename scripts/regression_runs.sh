#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent directory of scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Set the virtualenv path
VENV_PATH="$HOME/.power-attention-regression-venv"

# Create virtualenv if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtualenv at $VENV_PATH"
    python -m venv "$VENV_PATH"
fi

# Activate virtualenv
source "$VENV_PATH/bin/activate"

# Force reinstall requirements
echo "Force reinstalling requirements..."
pip install -r requirements.txt

# CD into training directory (now this will always work)
cd train || exit 1

# Check if git repo is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Git repository is not clean. Please commit or stash changes before running."
    exit 1
fi
# Track the commit hash
COMMIT_HASH=$(git rev-parse HEAD)

# Current date and time
CURRENT_TIME=$(date +"%Y%m%d%H%M")

# Run sequence of training runs with different hyperparameters
python train.py --run_name=regressions/default/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/largectx/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --batch_size=2 --block_size=16384 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p1_att/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=1 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p2_att/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=2 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p8_att/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=8 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p2_att_largectx/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=1000 --attention_kernel=power --degree=2 --batch_size=2 --block_size=16384 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p1/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=1 --chunk_size=128 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p2/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=2 --chunk_size=1024 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p1_largectx/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=1 --batch_size=2 --block_size=16384 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128
python train.py --run_name=regressions/p2_largectx/${CURRENT_TIME}_${COMMIT_HASH} --max_iters=5000 --attention_kernel=power --degree=2 --batch_size=2 --block_size=16384 --gradient_accumulation_steps=1 --n_layer=3 --n_head=2 --n_embd=128


# Deactivate virtualenv
deactivate

echo "Training sequence complete!" 