import os
from pathlib import Path

from dry_pipe import TaskConf


this_python_root = Path(__file__).parent.parent


def narval_task_conf(sbatch_options):
    remote_login = os.environ["REMOTE_LOGIN"]

    user = remote_login.split("@")[0].strip()

    remote_base_dir = f"/home/{user}/projects/def-marechal"

    remote_pipeline_base_dir = f"{remote_base_dir}/pipelines-work-dir"

    return TaskConf(
        executer_type="slurm",
        slurm_account=os.environ["SLURM_ACCOUNT"],
        sbatch_options=sbatch_options,
        extra_env={
            "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6",
            "DRYPIPE_TASK_DEBUG": "True",
            "PYTHONPATH": f"$__pipeline_instance_dir/external-file-deps{this_python_root}",
            "TASK_VENV": f"{remote_base_dir}/programs/colabfold_af2.3.2_env",
            "remote_base_dir": remote_base_dir,
            "collabfold_db": f"{remote_base_dir}/programs/colabfold_db",
            "HOME": "$__task_output_dir/fake_home"
        },
        ssh_remote_dest=f"{remote_login}:{remote_pipeline_base_dir}",
        python_bin=f"{remote_base_dir}/programs/colabfold_af2.3.2_env/bin/python3",
        run_as_group=os.environ["SLURM_ACCOUNT"]
    )

def gh_task_conf(sbatch_options):

    remote_pipeline_base_dir = "/tank/maxl"

    programs_base_dir = "/home/def-marechal/programs"

    return TaskConf(
        executer_type="slurm",
        slurm_account=None,
        sbatch_options=sbatch_options + ["--nodelist=gh1301  -p c-gh"],
        extra_env={
            "PYTHONPATH": f"$__pipeline_instance_dir/external-file-deps{this_python_root}",
            "python_bin": f"{programs_base_dir}/conda/envs/openfold_env/bin/python3",
            "remote_base_dir": remote_pipeline_base_dir,
            "collabfold_db": "/tank/jflucier/mmseqs_dbs",
            "OPENFOLD_HOME": f"{programs_base_dir}/openfold",
            "PATH": f"{programs_base_dir}/conda/envs/openfold_env/bin:{programs_base_dir}/MMseqs2/build/bin:$PATH",
            "HOME": "$__task_output_dir/fake_home",

            "DRYPIPE_TASK_DEBUG": "True",
            "CUDA_LAUNCH_BLOCKING": "1"
        },
        ssh_remote_dest=f"gh1301:{remote_pipeline_base_dir}",
        python_bin="/home/def-marechal/programs/conda/envs/openfold_env/bin/python3",
        run_as_group="def-marechal"
    )
