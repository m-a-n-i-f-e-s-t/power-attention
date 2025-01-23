#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent directory of scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

CURRENT_COMMIT=$(git rev-parse --short HEAD)

# Check if we're running in GitHub Actions
if [ -n "$GITHUB_ACTIONS" ]; then
    LOGGING_FLAG="--disable_logging=True"
else
    LOGGING_FLAG=""
fi

cd train
python train.py --batch_size=1 --block_size=32 --max_iters=100 --run_name=ci/sdpa/$CURRENT_COMMIT --data_dir=/shared/datasets/ngpt_owt $LOGGING_FLAG
python train.py --attention_kernel=power --chunk_size=32 --batch_size=1 --block_size=128 --max_iters=100 --run_name=ci/power/$CURRENT_COMMIT --data_dir=/shared/datasets/ngpt_owt $LOGGING_FLAG
python train.py --attention_kernel=power --batch_size=1 --block_size=32 --max_iters=100 --run_name=ci/power_att/$CURRENT_COMMIT --data_dir=/shared/datasets/ngpt_owt $LOGGING_FLAG