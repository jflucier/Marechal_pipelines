
# Adjust the following vars
BIO_TRACER_HOME=/home/maxl/BioTracer
DRYPIPE_HOME=/home/maxl/BioTracer/DryPipe


conda activate BioTracerEnv

export DRYPIPE_PIPELINE_INSTANCE_DIR=$1
export PYTHONPATH=$BIO_TRACER_HOME:$DRYPIPE_HOME

echo "DRYPIPE_PIPELINE_INSTANCE_DIR=$DRYPIPE_PIPELINE_INSTANCE_DIR"

