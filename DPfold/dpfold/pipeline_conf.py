
import os
from pathlib import Path


from dpfold.dag import colabfold_pipeline, parse_and_validate_input_files
from dry_pipe.pipeline import PipelineType


class DPFoldPT(PipelineType):

    def task_sort_key(self, task_key):
        if task_key == "cf-download-pdbs":
            return "000"
        elif task_key.startswith("cf-fold-array"):
            return "010"

        return "020"

    def array_grouper(self, task_key):

        if task_key.startswith("cf-fold."):
            return "cf-fold-array"

        return None

    def doc_root(self):
        return Path(__file__).parent


def gen_conf():

    def read_dir_from_env_var(name):

        v = os.environ.get(name)
        if v is None:
            raise Exception(f"missing env var {name}")
        if not os.path.exists(v):
            raise Exception(f"dir {v} specified by {name} must exist")

        return str(v)


    pipeline_run_site = read_dir_from_env_var("PIPELINE_INSTANCES_DIR")
    pipeline_code_dir = str(Path(__file__).parent.parent)
    dp_fold_instances_dir = str(Path(pipeline_run_site, "dp-fold"))
    Path(dp_fold_instances_dir).mkdir(exist_ok=True)

    common_schemas = {
        "schema": {
            "cc_cluster": ['enum', 'narval'],
            "cc_project": "string",
            "cc_allocation": "string"
        }
    }

    def dpfold_completion_func(pipeline_instance_dir):
        zipz = list(Path(pipeline_instance_dir, "output", "of-aggregate-report").glob("*.zip"))
        if len(zipz) > 0:
            csvs = Path(pipeline_instance_dir, "output", "of-aggregate-report").glob("*.csv")
            yield True, list(csvs) + zipz
        else:
            yield False, []

    def validate_dp_fold(pipeline_instance_dir):
        errors, samplesheet, multimers, _ = parse_and_validate_input_files(pipeline_instance_dir)

        return errors, None

    yield dp_fold_instances_dir, \
           DPFoldPT(
               "DPFold",
                pipeline=colabfold_pipeline(),
                validator=validate_dp_fold,
                spartan_schema=None,
                default_args={},
                complete_func=dpfold_completion_func,
                pre_run_filters=["cf-download-pdbs"]
           )
