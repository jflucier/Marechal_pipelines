#!/bin/bash

export DRYPIPE_PIPELINE_INSTANCE_DIR=$PWD/tiny-colabfold
export DRYPIPE_PIPELINE_GENERATOR="dpfold.dag:colabfold_pipeline"


. env.sh