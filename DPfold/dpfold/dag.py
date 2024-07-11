import glob
import os.path
from pathlib import Path
from dry_pipe import DryPipe

from dpfold.multimer import parse_multimer_list_from_samplesheet
from dpfold.multimer import file_path as multimer_code_file


@DryPipe.python_call()
def generate_pdb(samplesheet, multimer_name, __task_output_dir):
    multimer = parse_multimer_list_from_samplesheet(samplesheet, multimer_name)[0]
    multimer.generate_pdb(__task_output_dir)


@DryPipe.python_call()
def generate_fasta_colabfold(samplesheet, multimer_name, fa_out):
    multimer = parse_multimer_list_from_samplesheet(samplesheet, multimer_name)[0]
    return multimer.generate_fasta_colabfold(fa_out)


@DryPipe.python_call()
def generate_fasta_openfold(samplesheet, multimer_name, fa_out):
    multimer = parse_multimer_list_from_samplesheet(samplesheet, multimer_name)[0]
    return multimer.generate_fasta_openfold(fa_out)


def colabfold_analysis_script():

    af2_script = os.path.join(Path(__file__).parent.parent.parent, "AF2multimer-analysis", "colabfold_analysis.py")

    if not os.path.exists(af2_script):
        raise Exception(f"script not found:{af2_script}, git submodule not fetched.")

    return af2_script


@DryPipe.python_call()
def duplicate_stos(__task_output_dir):

    def sequence_basename_without_index_suffix(sequence_name):
        return "_".join(
            sequence_name.split("_")[:-1]
        )

    stos_by_basename = {
        sequence_basename_without_index_suffix(Path(sto).parent.name): sto
        for sto in glob.glob(os.path.join(__task_output_dir, "*", "uniprot_hits.sto"))
    }

    for d in glob.glob(os.path.join(__task_output_dir, "*", "uniref.a3m")):

        seq_subdir = Path(d).parent
        seq_name = seq_subdir.name
        expected_sto = os.path.join(seq_subdir, "uniprot_hits.sto")

        if not os.path.exists(expected_sto):

            basename = sequence_basename_without_index_suffix(seq_name)

            sto = stos_by_basename.get(basename)

            if sto is None:
                raise Exception(f"could not find uniprot_hits.sto for {seq_name}")

            os.symlink(sto, expected_sto)


def openfold_dag(dsl, list_of_multimers, samplesheet, task_conf):

    for multimer in list_of_multimers:
        fold_name = multimer.generate_openfold_fold_name()
        multimer_name = multimer.multimer_name()

        def fold_model(model):
            return f"""
              #!/usr/bin/bash                        
              set -ex
              
              echo "folding using model {model}"
              
              echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"
              echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
              echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
              
              nvidia-smi dmon \\
                --id=$SLURM_JOB_GPUS \\
                --select=u \\
                --options=T \\
                --filename=$__task_control_dir/nvidia-smi.log &
              
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

        yield dsl.task(
            key=f"analysis-openfold.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=task_conf
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

            echo "user home is $HOME"
            mkdir -p $HOME

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
                --raise_errors \\
                --uniprot_database_path $collabfold_db/uniprot/uniprot.fasta \\
                --jackhmmer_binary_path jackhmmer \\
                --cpus_per_task 32
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

    for match in dsl.query_all_or_nothing("analysis-openfold.*", state="ready"):
        yield dsl.task(
            key=f"analysis-openfold-array",
            task_conf=task_conf
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()


def collabfold_dag(dsl, list_of_multimers, samplesheet, collabfold_task_conf_func):

    colabfold_search_slurm_options = ["--time=24:00:00 --mem=40G --cpus-per-task=8"]

    for multimer in list_of_multimers:

        multimer_name = multimer.multimer_name()

        colabfold_search_task = dsl.task(
            key=f"colabfold_search.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=collabfold_task_conf_func(colabfold_search_slurm_options)
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

    for match in dsl.query_all_or_nothing("colabfold_search.*", state="ready"):
        yield dsl.task(
            key=f"colabfold-search-array",
            task_conf=collabfold_task_conf_func(colabfold_search_slurm_options)
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()

    for match in dsl.query_all_or_nothing("colabfold_search.*", state="completed"):
        for search_task in match.tasks:
            multimer_name = str(search_task.inputs.multimer_name)

            colabfold_batch_task = dsl.task(
                key=f"colabfold_batch.{multimer_name}",
                is_slurm_array_child=True
            ).inputs(
                a3m=search_task.outputs.a3m,
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

                export PATH=$remote_base_dir/programs/mmseqs/bin:$PATH

                echo "running colabfold fold"
                colabfold_batch \\
                  --use-gpu-relax --amber --num-relax 3 \\
                  --num-models 3 \\
                  --num-recycle 30 --recycle-early-stop-tolerance 0.5 \\
                  --model-type auto \\
                  --data $collabfold_db \\
                  $a3m \\
                  $__task_output_dir

                echo "running AF2multimer-analysis"
                mkdir -p $__task_output_dir/unrelaxed
                mv $__task_output_dir/*unrelaxed_* $__task_output_dir/unrelaxed/                                

                $python_bin -u $colabfold_analysis_script --pred_folder $__task_output_dir

                echo "done"

            """)()
            yield colabfold_batch_task

        for match in dsl.query_all_or_nothing("colabfold_batch.*", state="ready"):
            yield dsl.task(
                key=f"colabfold-batch-array",
                task_conf=["--time=8:00:00 --mem=120G --cpus-per-task=12 --gpus-per-node=1"]
            ).slurm_array_parent(
                children_tasks=match.tasks
            )()

    for _ in dsl.query_all_or_nothing("colabfold_batch.*"):
        # zip results
        yield dsl.task(
            key="zip_results",
            task_conf=collabfold_task_conf_func(["--time=1:00:00 --cpus-per-task=1"])
        ).calls("""
            #!/usr/bin/bash
            set -xe

            zip -r all_resuls.zip $__pipeline_instance_dir/output/colabfold_batch.* 
        """)()


def combined_pipeline_dag(dsl, openfold_task_conf_func, collabfold_task_conf_func):

    samplesheet = os.path.join(dsl.pipeline_instance_dir(), "samplesheet.tsv")

    multimers = parse_multimer_list_from_samplesheet(samplesheet)

    def is_long_sequence(multimer):
        return multimer.sequence_length() > 2700

    long_multimers = [
        m for m in multimers if is_long_sequence(m)
    ]

    short_multimers = [
        m for m in multimers if not is_long_sequence(m)
    ]

    yield from collabfold_dag(dsl, short_multimers, samplesheet, collabfold_task_conf_func)

    yield from openfold_dag(dsl, long_multimers, samplesheet, openfold_task_conf_func)

    #for match_openfold_tasks in dsl.query_all_or_nothing("analysis-openfold.*"):
    #    for match_collabfold_tasks in dsl.query_all_or_nothing("colabfold_batch.*"):
    #        pass




def colabfold_pipeline():

    from dpfold.task_confs import narval_task_conf

    def p(dsl):
        samplesheet = os.path.join(dsl.pipeline_instance_dir(), "samplesheet.tsv")

        multimers = parse_multimer_list_from_samplesheet(samplesheet)

        yield from collabfold_dag(dsl, multimers, samplesheet, narval_task_conf)

    return DryPipe.create_pipeline(p)


def openfold_pipeline():

    from dpfold.task_confs import big_gpu_task_conf

    def p(dsl):
        samplesheet = os.path.join(dsl.pipeline_instance_dir(), "samplesheet.tsv")

        multimers = parse_multimer_list_from_samplesheet(samplesheet)

        yield from collabfold_dag(dsl, multimers, samplesheet, big_gpu_task_conf)

    return DryPipe.create_pipeline(p)


def combined_pipeline():

    from dpfold.task_confs import big_gpu_task_conf, narval_task_conf

    return DryPipe.create_pipeline(
        lambda dsl: combined_pipeline_dag(dsl, big_gpu_task_conf, narval_task_conf)
    )
