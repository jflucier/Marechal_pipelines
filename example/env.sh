#!/bin/bash

cd ..

. venv/bin/activate

export DRYPIPE_PIPELINE_INSTANCE_DIR=$PWD/example/tiny
export PYTHONPATH=$PWD/DPfold

cd example

export COLLABFOLD_DB="/nfs3_ib/nfs-ip34/home/def-marechal/programs/colabfold_db"
export PIPELINE_REMOTE_BASE_DIR="/home/maxl/projects/def-marechal"
export REMOTE_LOGIN=maxl@narval.computecanada.ca
export SLURM_ACCOUNT=def-marechal


echo "run pipeline with: drypipe run --generator=dpfold:pipeline"