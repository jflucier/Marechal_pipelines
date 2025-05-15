#!/bin/bash


. venv/bin/activate


export PYTHONPATH=$(dirname $PWD)/DPfold


export DPFOLD_REMOTE_SSH_HOST_DIR=gh1301:/tank/maxl
export DRYPIPE_PIPELINE_GENERATOR="dpfold.dag:openfold_pipeline"
export DRYPIPE_PIPELINE_INSTANCE_DIR=$1

echo "pipeline: $DRYPIPE_PIPELINE_GENERATOR in $DRYPIPE_PIPELINE_INSTANCE_DIR"
echo "run with: drypipe run"



