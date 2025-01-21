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
pip install -r requirements.txt --force-reinstall

# CD into training directory (now this will always work)
cd train || exit 1

# Check if git repo is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Git repository is not clean. Please commit or stash changes before running."
    exit 1
fi
# Track the commit hash
COMMIT_HASH=$(git rev-parse HEAD)

# Run sequence of training runs with different hyperparameters
python train.py --run_name=regressions/default/$COMMIT_HASH --max_iters=5000
python train.py --run_name=regressions/largectx/$COMMIT_HASH --max_iters=5000 --batch_size=2 --block_size=16384
python train.py --run_name=regressions/p1_att/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=1
python train.py --run_name=regressions/p2_att/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=2
python train.py --run_name=regressions/p8_att/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=8
python train.py --run_name=regressions/p2_att_largectx/$COMMIT_HASH --max_iters=1000 --attention_kernel=power --degree=2 --batch_size=2 --block_size=16384
python train.py --run_name=regressions/p1/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=1 --chunk_size=128
python train.py --run_name=regressions/p2/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=2 --chunk_size=1024
python train.py --run_name=regressions/p1_largectx/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=1 --batch_size=2 --block_size=16384
python train.py --run_name=regressions/p2_largectx/$COMMIT_HASH --max_iters=5000 --attention_kernel=power --degree=2 --batch_size=2 --block_size=16384


# Deactivate virtualenv
deactivate

echo "Training sequence complete!" 