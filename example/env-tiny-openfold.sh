#!/bin/bash

export DRYPIPE_PIPELINE_INSTANCE_DIR=$PWD/tiny-openfold
export DRYPIPE_PIPELINE_GENERATOR="dpfold.dag:openfold_pipeline"

. env.sh

