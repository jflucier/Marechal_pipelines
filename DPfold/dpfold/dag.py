import os.path
from pathlib import Path

import requests

import pandas as pd

from dry_pipe import TaskConf, DryPipe

def collabfold_db():

    if "COLLABFOLD_DB" in os.environ:
        return os.environ["COLLABFOLD_DB"]

    return "/nfs3_ib/nfs-ip34/home/def-marechal/programs/colabfold_db"


def remote_base_dir():

    if "PIPELINE_REMOTE_BASE_DIR" in os.environ:
        d = os.environ["PIPELINE_REMOTE_BASE_DIR"]

        if d == "":
            raise Exception(f"PIPELINE_REMOTE_BASE_DIR can't be empty")

        return d

    raise Exception(f"Env var PIPELINE_REMOTE_BASE_DIR must be set")


task_conf = TaskConf(
    executer_type="process",
    extra_env={
        "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6"
    }
)


this_python_root = Path(__file__).parent.parent
def narval_task_conf():
    remote_login = os.environ["REMOTE_LOGIN"]

    return TaskConf(
        executer_type="slurm",
        slurm_account=os.environ["SLURM_ACCOUNT"],
        sbatch_options=["--time=0:5:00"],
        extra_env={
            "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6",
            "DRYPIPE_TASK_DEBUG": "True",
            "PYTHONPATH":
                f"{remote_base_dir()}/pipelines-work-dir/tiny/external-file-deps{this_python_root}",
            "TASK_VENV": f"{remote_base_dir()}/programs/colabfold_af2.3.2_env/bin/activate"
        },
        ssh_remote_dest=f"{remote_login}:{remote_base_dir()}/pipelines-work-dir",
        # implicit:
        python_bin=f"{remote_base_dir()}/programs/colabfold_af2.3.2_env/bin/python3"
    )

def mp2_task_conf():
    return TaskConf(
        executer_type="slurm",
        slurm_account="def-marechal",
        sbatch_options=["--time=0:25:00"],
        extra_env={
            "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6",
            "DRYPIPE_TASK_DEBUG": "True"
        },
        ssh_remote_dest="maxl@mp2.ccs.computecanada.ca:/home/maxl/test-tiny",
        # implicit:
        python_bin="/home/maxl/venv-marechal/bin/python3"
    )


# not required for now. Will see later if work without network
@DryPipe.python_call()
def generate_pdb(row, __task_output_dir):
    pdb_dir = os.path.join(__task_output_dir, "pdb")
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    prot_nbr = 1
    while f"protein{prot_nbr}_PDB" in row.index:
        if not pd.isna(row[f"protein{prot_nbr}_PDB"]):
            pdb_str = row[f"protein{prot_nbr}_PDB"]
            pdb_list = pdb_str.split(",")
            for pdb in pdb_list:
                # download pdb if pdb id is provided
                pdb_out = os.path.join(pdb_dir, f"{pdb.lower()}.cif")
                if os.path.exists(pdb_out):
                    print(f"{pdb_out} already found. No need to re-download")
                else:
                    print(f"Downloading and generating pdb: {pdb}")
                    URL = f"https://files.rcsb.org/download/{pdb}.cif"
                    response = requests.get(URL)
                    with open(pdb_out, 'w') as out:
                        out.write(response.text)
                        # modify cif file to include new section. Change name to pdb
                        # https://github.com/sokrypton/ColabFold/issues/177
                        out.write('#\n')
                        out.write('loop_\n')
                        out.write('_pdbx_audit_revision_history.ordinal\n')
                        out.write('_pdbx_audit_revision_history.data_content_type\n')
                        out.write('_pdbx_audit_revision_history.major_revision\n')
                        out.write('_pdbx_audit_revision_history.minor_revision\n')
                        out.write('_pdbx_audit_revision_history.revision_date\n')
                        out.write('1 \'Structure model\' 1 0 1971-01-01\n')
                        out.write('#\n')

        prot_nbr = prot_nbr + 1

def get_row(samplesheet, multimer_name):
    folds = pd.read_csv(samplesheet, sep='\t', index_col="multimer_name")
    for index, row in folds.iterrows():
        if index != multimer_name:
            continue

        return row

@DryPipe.python_call()
def generate_fasta(samplesheet, multimer_name, fa_out):

    row = get_row(samplesheet, multimer_name)
    prot_nbr = 1
    fa_header = ""
    fa_seq = []
    while f"protein{prot_nbr}_name" in row.index:
        if not pd.isna(row[f"protein{prot_nbr}_name"]):
            p_name = row[f"protein{prot_nbr}_name"]
            p_nbr = int(row[f"protein{prot_nbr}_nbr"])
            p_seq = row[f"protein{prot_nbr}_seq"]
            fa_header = fa_header + f"{p_name}_{p_nbr}_"
            fa_seq.extend([p_seq] * p_nbr)
        prot_nbr = prot_nbr + 1

    fa_header = fa_header[:-1]
    seq = ":".join(fa_seq)
    with open(fa_out, 'w') as f:
        f.write(f">{fa_header}\n")
        f.write(f"{seq}\n")


def dag_gen(dsl):

    samplesheet = os.path.join(dsl.pipeline_instance_dir(), "samplesheet.tsv")

    folds = pd.read_csv(samplesheet, sep='\t', index_col="multimer_name")

    for index, row in folds.iterrows():
        # generate msa alignment
        # on cpu compute nodes

        colabfold_search_task = dsl.task(
            key=f"colabfold_search.{index}",
            is_slurm_array_child=True,
            task_conf=narval_task_conf()
        ).inputs(
            samplesheet=dsl.file(samplesheet),
            multimer_name=index,
            db=collabfold_db(),
            code_dep=dsl.file(__file__)
        ).outputs(
            fa_out=dsl.file(f'fold.fa'),
            a3m=dsl.file(f'0.a3m')
        ).calls(
            generate_fasta
        ).calls("""
            #!/usr/bin/bash
            
            set -e
            
            module load gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 mmseqs2/14-7e284 hh-suite/3.3.0 hmmer/3.2.1
                        
            source $TASK_VENV/bin/activate
            
            export IN=$fa_out
            export OUT=$__task_output_dir
            export DOWNLOAD_DIR=$db
            
            echo "running colabfold search"
            colabfold_search \
            --threads 8 --use-env 1 --db-load-mode 0 \
            --mmseqs mmseqs \
            --db1 ${DOWNLOAD_DIR}/uniref30_2302_db \
            --db2 ${DOWNLOAD_DIR}/pdb100_230517 \
            --db3 ${DOWNLOAD_DIR}/colabfold_envdb_202108_db \
            ${IN} ${DOWNLOAD_DIR} ${OUT}
            
            echo "done"

        """)()
        yield colabfold_search_task

    for match in dsl.query_all_or_nothing("colabfold_search.*", state="ready"):
        yield dsl.task(
            key=f"colabfold-search-array",
            task_conf=narval_task_conf()
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()

        for index, row in folds.iterrows():
            # generate msa aligmenent
            # on gpu compute nodes
            colabfold_batch_task = dsl.task(
                key=f"colabfold_batch.{index}",
                is_slurm_array_child=True,
                task_conf=narval_task_conf()
            ).inputs(
                a3m=colabfold_search_task.outputs.a3m,
                db=collabfold_db()
            ).outputs(
                relaxed_pdb=dsl.file(f'fold.fa'),
                unrelaxed_pdb=dsl.file(f'0.a3m')
            ).calls("""
                #!/usr/bin/bash
    
                set -e
    
                module load gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 mmseqs2/14-7e284 hh-suite/3.3.0 hmmer/3.2.1
    
                source $TASK_VENV/bin/activate
                
                export TF_FORCE_UNIFIED_MEMORY="1"
                export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
                export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
                export TF_FORCE_GPU_ALLOW_GROWTH="true"
    
                export IN=$a3m
                export OUT=$__task_output_dir
                export DOWNLOAD_DIR=$db
    
                echo "running colabfold fold"
                colabfold_batch \
                --use-gpu-relax --amber --num-relax 3 \
                --num-models 3 \
                --num-recycle 30 --recycle-early-stop-tolerance 0.5 \
                --model-type auto \
                --data $DOWNLOAD_DIR \
                ${IN} \
                ${OUT}
                
                echo "running AF2multimer-analysis"
                mkdir -p $__task_output_dir/unrelaxed
                mv $__task_output_dir/*unrelaxed_* $__task_output_dir/unrelaxed/
                # to change
                python $__pipeline_code_dir/../AF2multimer-analysis/colabfold_analysis.py $__task_output_dir
    
                echo "done"
    
            """)()
            yield colabfold_batch_task

    for _ in dsl.query_all_or_nothing("colabfold_batch.*"):
        # zip results
        yield dsl.task(
            key="zip_results"
        ).calls("""
            #!/usr/bin/bash
            set -xe
            
            zip -r all_resuls.zip $__pipeline_instance_dir/output/colabfold_batch.* 
        """)()

# if __name__ == '__main__':
