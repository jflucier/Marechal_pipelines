from dpfold.dag import dag_gen
from dry_pipe import DryPipe


def pipeline():
    return DryPipe.create_pipeline(dag_gen)