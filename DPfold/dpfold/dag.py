import os.path

from pathlib import Path

import requests

import pandas as pd

from dpfold.multimer import parse_multimer_list_from_samplesheet
from dpfold.multimer import file_path as multimer_code_file
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


this_python_root = Path(__file__).parent.parent

def colabfold_analysis_script():

    af2_script = os.path.join(this_python_root.parent, "AF2multimer-analysis", "colabfold_analysis.py")

    if not os.path.exists(af2_script):
        raise Exception(f"script not found:{af2_script}, git submodule not fetched.")

    return af2_script


task_conf = TaskConf(
    executer_type="process",
    extra_env={
        "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6"
    }
)


def narval_task_conf():
    remote_login = os.environ["REMOTE_LOGIN"]

    remote_pipeline_base_dir = f"{remote_base_dir()}/pipelines-work-dir"

    return TaskConf(
        executer_type="slurm",
        slurm_account=os.environ["SLURM_ACCOUNT"],
        sbatch_options=["--time=6:00:00 --mem=40G --cpus-per-task=8"],
        extra_env={
            "MUGQIC_INSTALL_HOME": "/cvmfs/soft.mugqic/CentOS6",
            "DRYPIPE_TASK_DEBUG": "True",
            "PYTHONPATH":
                f"{remote_pipeline_base_dir}/tiny/external-file-deps{this_python_root}",
            "TASK_VENV": f"{remote_base_dir()}/programs/colabfold_af2.3.2_env",
            "remote_base_dir": remote_base_dir(),
            "collabfold_db": collabfold_db()
        },
        ssh_remote_dest=f"{remote_login}:{remote_pipeline_base_dir}",
        # implicit:
        python_bin=f"{remote_base_dir()}/programs/colabfold_af2.3.2_env/bin/python3"
    )

def big_gpu_task_conf():

    remote_pipeline_base_dir = "/tank/maxl"
    programs_base_dir = "/home/def-marechal/programs"

    return TaskConf(
        executer_type="slurm",
        slurm_account=None,
        sbatch_options=[
            "--time=24:00:00 --gpus-per-node=1 --cpus-per-task=64 --mem=440G",
            "--nodelist=gh1301  -p c-gh"
        ],
        extra_env={
            "DRYPIPE_TASK_DEBUG": "True",
            "PYTHONPATH": f"$__pipeline_instance_dir/external-file-deps{this_python_root}",
            "python_bin": f"{programs_base_dir}/conda/envs/openfold_env/bin/python3",
            "remote_base_dir": "/tank/maxl",
            "collabfold_db": "/tank/jflucier/mmseqs_dbs",
            "OPENFOLD_HOME": f"{programs_base_dir}/openfold",
            "PATH": f"{programs_base_dir}/conda/envs/openfold_env/bin:{programs_base_dir}/MMseqs2/build/bin:$PATH"
        },
        ssh_remote_dest=f"gh1301:{remote_pipeline_base_dir}",
        python_bin="/home/def-marechal/programs/conda/envs/openfold_env/bin/python3"
    )



# not required for now. Will see later if work without network
@DryPipe.python_call()
def generate_pdb(samplesheet, multimer_name, __task_output_dir):
    multimer = parse_multimer_list_from_samplesheet(samplesheet, multimer_name)[0]
    multimer.generate_pdb(__task_output_dir)


@DryPipe.python_call()
def generate_fasta_colabfold(samplesheet, multimer_name, fa_out):
    multimer = parse_multimer_list_from_samplesheet(samplesheet, multimer_name)[0]
    return multimer.generate_fasta_colabfold(fa_out)


@DryPipe.python_call()
def generate_fasta_openfold(fa_out, samplesheet, multimer_name):
    multimer = parse_multimer_list_from_samplesheet(samplesheet, multimer_name)[0]
    return multimer.generate_fasta_openfold(fa_out)


def dag_gen(dsl):

    samplesheet = os.path.join(dsl.pipeline_instance_dir(), "samplesheet.tsv")

    multimers = parse_multimer_list_from_samplesheet(samplesheet)

    def requires_big_gpu_node(multimer):
        return False
        #return multimer.sequence_length() > 2700

    def openfold_analysis(multimer):

        fold_name = multimer.generate_openfold_fold_name()
        multimer_name = multimer.multimer_name()

        def fold_model(model):
            return f"""
              #!/usr/bin/bash                        
              set -ex
                          
              echo "folding using model {model}"
              cd $OPENFOLD_HOME
                            
              $python_bin -u $OPENFOLD_HOME/run_pretrained_openfold.py \\
                $__task_output_dir \\
                $collabfold_db/pdb_mmcif/mmcif_files \\
                --use_precomputed_alignments $__task_output_dir \\
                --config_preset "model_{model}_multimer_v3" \\
                --model_device "cuda:0" \\
                --output_dir $__task_output_dir \\
                --save_outputs
            
              echo "generate JSON for model {model}"
              $python_bin -u $OPENFOLD_HOME/scripts/pickle_to_json.py \\
                --model_pkl $__task_output_dir/predictions/{fold_name}_model_{model}_multimer_v3_output_dict.pkl \\
                --output_dir $__task_output_dir/predictions/ \\
                --basename "{fold_name}" \\
                --model_nbr {model}                
            """

        return dsl.task(
            key=f"analysis-openfold.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=big_gpu_task_conf()
        ).inputs(
            samplesheet=dsl.file(samplesheet),
            multimer_name=multimer_name,
            code_dep1=dsl.file(__file__),
            code_dep2=dsl.file(multimer_code_file()),
            fold_name=fold_name,
            colabfold_analysis_script=dsl.file(colabfold_analysis_script())
        ).outputs(
            fa_out=dsl.file(f'{fold_name}.fa'),
            analysis_zip=dsl.file(f'{fold_name}.zip'),
        ).calls(
            generate_fasta_openfold
        ).calls("""
            #!/usr/bin/bash                        
            set -e
            
            cd $OPENFOLD_HOME
                    
            $python_bin -u ${OPENFOLD_HOME}/scripts/precompute_alignments_mmseqs.py \\
               --threads 64 \\
               --hhsearch_binary_path hhsearch \\
               --pdb70 $collabfold_db/pdb100 \\
               --env_db colabfold_envdb_202108_db \\
               $fa_out \\
               $collabfold_db \\
               uniref30_2302_db \\
               $__task_output_dir \\
               mmseqs
        """).calls("""
            #!/usr/bin/bash                        
            set -e
            cd $OPENFOLD_HOME
                        
            echo "running uniprot alignment on $__task_output_dir"
            $python_bin -u ${OPENFOLD_HOME}/scripts/precompute_alignments.py \\
                $__task_output_dir \\
                $__task_output_dir \\
                --uniprot_database_path $collabfold_db/uniprot/uniprot.fasta \\
                --jackhmmer_binary_path jackhmmer \\
                --cpus_per_task 32                
                
            ls $__task_output_dir/uniprot_hits.sto
                
            """
        ).calls(
            fold_model(1)
        ).calls(
            fold_model(2)
        ).calls(
            fold_model(3)
        ).calls(
            """
            #!/usr/bin/bash
            set -e
            echo "generating coverage plots"
            $python_bin -u ${OPENFOLD_HOME}/scripts/generate_coverage_plot.py \\
              --input_pkl $__task_output_dir/${fold_name}_model_1_multimer_v3_feature_dict.pickle \\
              --output_dir $__task_output_dir/predictions/ \\
              --basename "${fold_name}_multimer_v3_relaxed"            
            """
        ).calls(
            """
            #!/usr/bin/bash                        
            set -e            
            echo "running AF2multimer-analysis on $__task_output_dir/predictions/"
            touch $__task_output_dir/predictions/${fold_name}.done.txt
            mkdir -p $__task_output_dir/predictions/unrelaxed
            mv $__task_output_dir/predictions/*unrelaxed.pdb $__task_output_dir/predictions/unrelaxed/
            $python_bin -u $colabfold_analysis_script $__task_output_dir/predictions            
            """
        ).calls(
            """
            #!/usr/bin/bash                        
            set -e            
            echo "generating PAE, plDDT plots and JSON files"
            $python_bin -u ${OPENFOLD_HOME}/scripts/generate_pae_plddt_plot.py \\
              --fasta $__task_output_dir/${fold_name}.fa \\
              --model1_pkl $__task_output_dir/predictions/${fold_name}_model_1_multimer_v3_output_dict.pkl \\
              --model2_pkl $__task_output_dir/predictions/${fold_name}_model_2_multimer_v3_output_dict.pkl \\
              --model3_pkl $__task_output_dir/predictions/${fold_name}_model_3_multimer_v3_output_dict.pkl \\
              --output_dir $__task_output_dir/predictions/ \\
              --basename "${fold_name}" \\
              --interface $__task_output_dir/predictions/predictions_analysis/interfaces.csv            
              
            cd $__task_output_dir
            zip -r $analysis_zip * -x "*.pkl" "*.pickle"              
            """
        )()

    for multimer in multimers:

        multimer_name = multimer.multimer_name()

        if requires_big_gpu_node(multimer):
            yield openfold_analysis(multimer)
        else:

            colabfold_search_task = dsl.task(
                key=f"colabfold_search.{multimer_name}",
                is_slurm_array_child=True,
                task_conf=narval_task_conf()
            ).inputs(
                samplesheet=dsl.file(samplesheet),
                multimer_name=multimer_name,
                code_dep1=dsl.file(__file__),
                code_dep2=dsl.file(multimer_code_file())
            ).outputs(
                fa_out=dsl.file(f'fold.fa'),
                a3m=dsl.file(f'0.a3m')
            ).calls(
                generate_fasta_colabfold
            ).calls("""
                #!/usr/bin/bash
                
                set -e
                
                module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 hh-suite/3.3.0 hmmer/3.2.1
    
                TE=$TASK_VENV/bin/activate                      
                echo "will activate env: $TE"
                source $TE                                                
                            
                export PATH=$remote_base_dir/programs/mmseqs/bin:$PATH
                
                echo "running colabfold search"
                colabfold_search \\
                  --threads 8 --use-env 1 --db-load-mode 0 \\
                  --mmseqs mmseqs \\
                  --db1 $collabfold_db/uniref30_2302_db \\
                  --db2 $collabfold_db/pdb100_230517 \\
                  --db3 $collabfold_db/colabfold_envdb_202108_db \\
                  $fa_out $collabfold_db $__task_output_dir
                
                echo "done"
    
            """)()
            yield colabfold_search_task

    for match in dsl.query_all_or_nothing("analysis-openfold.*", state="ready"):
        yield dsl.task(
            key=f"analysis-openfold-array",
            task_conf=big_gpu_task_conf()
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()

    for match in dsl.query_all_or_nothing("colabfold_search.*", state="ready"):

        yield dsl.task(
            key=f"colabfold-search-array",
            task_conf=narval_task_conf()
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()

    for match in dsl.query_all_or_nothing("colabfold_search.*", state="completed"):
        for search_task in match.tasks:

            multimer_name = str(search_task.inputs.multimer_name)

            colabfold_batch_task = dsl.task(
                key=f"colabfold_batch.{multimer_name}",
                is_slurm_array_child=True,
                task_conf=narval_task_conf()
            ).inputs(
                a3m=search_task.outputs.a3m,
                db=collabfold_db(),
                colabfold_analysis_script=dsl.file(colabfold_analysis_script()),
                code_dep1=dsl.file(__file__),
                code_dep2=dsl.file(multimer_code_file())
            ).outputs(
                relaxed_pdb=dsl.file(f'fold.fa'),
                unrelaxed_pdb=dsl.file(f'0.a3m')
            ).calls("""
                #!/usr/bin/bash
    
                set -e
    
                module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 hh-suite/3.3.0 hmmer/3.2.1
    
                source $TASK_VENV/bin/activate
                
                export TF_FORCE_UNIFIED_MEMORY="1"
                export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
                export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
                export TF_FORCE_GPU_ALLOW_GROWTH="true"
    
                export IN=$a3m
                export OUT=$__task_output_dir
                export DOWNLOAD_DIR=$db
                
                export PATH=$remote_base_dir/programs/mmseqs/bin:$PATH
    
                echo "running colabfold fold"
                colabfold_batch \\
                  --use-gpu-relax --amber --num-relax 3 \\
                  --num-models 3 \\
                  --num-recycle 30 --recycle-early-stop-tolerance 0.5 \\
                  --model-type auto \\
                  --data $DOWNLOAD_DIR \\
                  ${IN} \\
                  ${OUT}
                
                echo "running AF2multimer-analysis"
                mkdir -p $__task_output_dir/unrelaxed
                mv $__task_output_dir/*unrelaxed_* $__task_output_dir/unrelaxed/                                
                
                python -u $colabfold_analysis_script $__task_output_dir
    
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
