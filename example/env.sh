#!/bin/bash

cd ..

. venv/bin/activate

export PYTHONPATH=$PWD/DPfold
export REMOTE_LOGIN=maxl@narval.computecanada.ca
export SLURM_ACCOUNT=def-marechal

cd -

echo "pipeline: $DRYPIPE_PIPELINE_GENERATOR in $DRYPIPE_PIPELINE_INSTANCE_DIR"
echo "run with: drypipe run"