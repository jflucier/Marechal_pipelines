import os.path
from itertools import groupby

from dry_pipe import TaskConf

samplesheet = "test.samplesheet.tsv"
colabfold_db = "/project/def-marechal/programs/colabfold_db"

task_conf = TaskConf(
    executer_type="process",
    extra_env={
        "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6"
    }
    # implicit:
    # python_bin="/home/maxl/miniconda3/envs/Genomics/bin/python"
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

@DryPipe.python_call()
def generate_fasta(row):
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
    folds = pd.read_csv(samplesheet, sep='\t', index_col="multimer_name")

    for index, row in folds.iterrows():
        # generate msa aligmenent
        # on cpu compute nodes
        colabfold_search_task = dsl.task(
            key=f"colabfold_search.{index}"
        ).inputs(
            row=row,
            db=colabfold_db
        ).outputs(
            fa_out=dsl.file(f'fold.fa'),
            a3m=dsl.file(f'0.a3m')
        ).calls(
            generate_fasta
        ).calls("""
            #!/usr/bin/bash
            
            set -e
            
            module load gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 mmseqs2/14-7e284 hh-suite/3.3.0 hmmer/3.2.1
            
            #AF_VERSION=2.3.1
            AF_VERSION=2.3.2            
            ENV=/home/jflucier/projects/def-marechal/programs/colabfold_af${AF_VERSION}_env/bin/activate
            source ${ENV}
            
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

        # generate msa aligmenent
        # on gpu compute nodes
        colabfold_batch_task = dsl.task(
            key=f"colabfold_batch.{index}"
        ).inputs(
            a3m=colabfold_search_task.outputs.a3m,
            db=colabfold_db
        ).outputs(
            relaxed_pdb=dsl.file(f'fold.fa'),
            unrelaxed_pdb=dsl.file(f'0.a3m')
        ).calls("""
            #!/usr/bin/bash

            set -e

            module load gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 mmseqs2/14-7e284 hh-suite/3.3.0 hmmer/3.2.1

            #AF_VERSION=2.3.1
            AF_VERSION=2.3.2            
            ENV=/home/jflucier/projects/def-marechal/programs/colabfold_af${AF_VERSION}_env/bin/activate
            source ${ENV}
            
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
