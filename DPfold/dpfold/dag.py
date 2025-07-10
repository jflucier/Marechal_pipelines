import glob
import os.path
from pathlib import Path
import json
from dry_pipe import DryPipe

from dpfold import colabfold_analysis
from dpfold.multimer import parse_multimer_list_from_samplesheet
from dpfold.multimer import file_path as multimer_code_file


def parse_and_validate_input_files(pipeline_instance_dir):

    samplesheet = os.path.join(pipeline_instance_dir, "samplesheet.tsv")
    pipeline_instance_args_file = os.path.join(pipeline_instance_dir, "args.json")

    if os.path.exists(pipeline_instance_args_file):
        with open(pipeline_instance_args_file) as f:
            pipeline_instance_args = json.loads(f.read())
    else:
        pipeline_instance_args = None

    samplesheet_parse_exception = None

    try:
        multimers = parse_multimer_list_from_samplesheet(samplesheet)
    except Exception as e:
        multimers = None
        samplesheet_parse_exception = e

    def go():

        if pipeline_instance_args is None:
            yield "MISSING_ARG_FILE", f"could not find pipeline_instance_args_file {pipeline_instance_args_file}"

        if "cc_project" not in pipeline_instance_args:
            yield "MISSING_ARG", "'cc_group' must be specified in Pipeline Args"

        if samplesheet_parse_exception is not None:
            yield "MISSING_ARG_FILE", f"could not parse {samplesheet}"


    return dict(go()), samplesheet, multimers, pipeline_instance_args


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


@DryPipe.python_call()
def generate_aggregate_report(__pipeline_instance_dir, interfaces_csv, summary_csv, contacts_csv):

    def gen_interfaces_lines():

        yield "fold_name,complex_name,model_num,pdockq,ncontacts,plddt_min,plddt_avg,plddt_max,pae_min,pae_avg,pae_max,distance_avg"

        for interfaces in Path(__pipeline_instance_dir, "output").glob("*/interfaces.csv"):
            with open(interfaces) as f_interfaces:
                for line in f_interfaces:
                    yield line.strip()

    def gen_summary_lines():
        yield "fold_name,complex_name,avg_n_models,max_n_models,num_contacts_with_max_n_models,num_unique_contacts,best_model_num,best_pdockq,best_plddt_avg,best_pae_avg"

        for summary in Path(__pipeline_instance_dir, "output").glob("*/summary.csv"):
            with open(summary) as f_summary:
                for line in f_summary:
                    yield line.strip()

    def gen_contact_lines():

        yield "fold_name,complex_name,model_num,aa1_chain,aa1_index,aa1_type,aa1_plddt,aa2_chain,aa2_index,aa2_type,aa2_plddt,pae,min_distance"

        for contact in Path(__pipeline_instance_dir, "output").glob("*/contacts.csv"):
            with open(contact) as f_contact:
                for line in f_contact:
                    yield line.strip()

    def flush_lines_into(lines, file):
        with open(file, "w") as f:
            for line in lines:
                f.write(line)
                f.write("\n")

    flush_lines_into(gen_interfaces_lines(), interfaces_csv)

    flush_lines_into(gen_summary_lines(), summary_csv)

    flush_lines_into(gen_contact_lines(), contacts_csv)


def openfold_dag(dsl, list_of_multimers, samplesheet, task_conf_func):

    slurm_options_align = ["--time=4:00:00 --cpus-per-task=8 --mem=100G"]

    slurm_options_fold = ["--time=24:00:00 --gpus-per-node=1 --cpus-per-task=8 --mem=50G"]

    for multimer in list_of_multimers:
        fold_name = multimer.generate_openfold_fold_name()
        multimer_name = multimer.multimer_name()

        yield dsl.task(
            key=f"of-align.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=task_conf_func(slurm_options_align)
        ).inputs(
            samplesheet=dsl.file(samplesheet),
            multimer_name=multimer_name,
            code_dep1=dsl.file(__file__),
            code_dep2=dsl.file(multimer_code_file()),
            fold_name=fold_name,
            colabfold_analysis_script=dsl.file(colabfold_analysis.code_path())
        ).outputs(
            fa_out=dsl.file(f'{fold_name}.fa')
        ).calls(
            generate_fasta_openfold
        ).calls("""
            #!/usr/bin/bash                        
            set -ex

            echo "user home is $HOME"
            mkdir -p $HOME

            cd $OPENFOLD_HOME

            $python_bin -u ${OPENFOLD_HOME}/scripts/precompute_alignments_mmseqs.py \\
               --threads 7 \\
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
            set -ex
            cd $OPENFOLD_HOME

            echo "running uniprot alignment on $__task_output_dir"
            $python_bin -u ${OPENFOLD_HOME}/scripts/precompute_alignments.py \\
                $__task_output_dir \\
                $__task_output_dir \\
                --raise_errors \\
                --uniprot_database_path $collabfold_db/uniprot/uniprot.fasta \\
                --jackhmmer_binary_path jackhmmer \\
                --cpus_per_task 7
            """
        ).calls(
            duplicate_stos
        )()

        def fold_model(model):
            return f"""
              #!/usr/bin/bash                        
              set -ex

              echo "folding using model {model}"
              
              echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
              echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"                          

              in_dir=$__pipeline_instance_dir/output/of-align.${{multimer_name}}
              
              echo "OPENFOLD_HOME=$OPENFOLD_HOME"              
              
              cd $OPENFOLD_HOME
                                                    
              echo "PATH: $PATH:"
              
              export PYTORCH_MEM_HISTORY_DUMP=$__task_output_dir/mem_dump_for_model_{model}.pickle
              
              if [ "$protein_count" = "1" ]; then
                  config_preset="model_{model}_ptm"                  
                  model_pkl=$__task_output_dir/predictions/${{fold_name}}_model_{model}_ptm_output_dict.pkl
              else
                  config_preset="model_{model}_multimer_v3"
                  model_pkl=$__task_output_dir/predictions/${{fold_name}}_model_{model}_multimer_v3_output_dict.pkl
              fi                            

              $python_bin -u $OPENFOLD_HOME/run_pretrained_openfold.py \\
                $in_dir \\
                $collabfold_db/pdb_mmcif/mmcif_files \\
                --use_precomputed_alignments $in_dir \\
                --config_preset $config_preset \\
                --model_device "cuda:$SLURM_JOB_GPUS" \\
                --output_dir $__task_output_dir \\
                --jax_param_path $OPENFOLD_HOME/openfold/resources/params/params_${{config_preset}}.npz \\
                --save_outputs

              echo "generate JSON for model {model}"
              $python_bin -u $OPENFOLD_HOME/scripts/pickle_to_json.py \\
                --model_pkl $model_pkl \\
                --output_dir $__task_output_dir/predictions/ \\
                --basename ${{fold_name}} \\
                --model_nbr {model}                
            """

        yield dsl.task(
            key=f"of-fold.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=task_conf_func(slurm_options_fold)
        ).inputs(
            fold_name=fold_name,
            multimer_name=multimer_name,
            protein_count=str(multimer.protein_count()),
            colabfold_analysis_script=dsl.file(colabfold_analysis.code_path())
        ).outputs(
            all_results=dsl.file_set("**/*"),
        ).calls(
            fold_model(1)
        ).calls(
            fold_model(2)
        ).calls(
            fold_model(3)
        ).calls(
            """
            #!/usr/bin/bash
            set -ex
            echo "generating coverage plots"
                        
            align_task_out=$__pipeline_instance_dir/output/of-align.${multimer_name}
            
            if [ "$protein_count" = "1" ]; then
            
               input_pkl=$__task_output_dir/${fold_name}_model_1_ptm_feature_dict.pickle
               plot_basename=$__task_output_dir/${fold_name}_relaxed
                                             
               model_1_pkl=$__task_output_dir/predictions/${fold_name}_model_1_ptm_output_dict.pkl
               model_2_pkl=$__task_output_dir/predictions/${fold_name}_model_2_ptm_output_dict.pkl
               model_3_pkl=$__task_output_dir/predictions/${fold_name}_model_3_ptm_output_dict.pkl
            else         
               input_pkl=$__task_output_dir/${fold_name}_model_1_multimer_v3_feature_dict.pickle
               plot_basename = $__task_output_dir/${fold_name}_multimer_v3_relaxed                              
                     
               model_1_pkl=$__task_output_dir/predictions/${fold_name}_model_1_multimer_v3_output_dict.pkl
               model_2_pkl=$__task_output_dir/predictions/${fold_name}_model_2_multimer_v3_output_dict.pkl
               model_3_pkl=$__task_output_dir/predictions/${fold_name}_model_3_multimer_v3_output_dict.pkl
            fi                            
            
            $python_bin -u ${OPENFOLD_HOME}/scripts/generate_coverage_plot.py \\
              --input_pkl $input_pkl \\
              --output_dir $__task_output_dir/predictions/ \\
              --basename $plot_basename
   
            echo "running AF2multimer-analysis on $__task_output_dir/predictions/"                        
            touch $__task_output_dir/predictions/${fold_name}.done.txt            
            mkdir -p $__task_output_dir/predictions/unrelaxed            
            mv $__task_output_dir/predictions/*unrelaxed.pdb $__task_output_dir/predictions/unrelaxed/ || true
            
            if [ "$protein_count" = "1" ]; then
              echo "will skip colabfold_analysis_script"
            else   
              $python_bin -u $colabfold_analysis_script \\
                 --pred_folder=$__task_output_dir/predictions \\
                 --out_folder=$__task_output_dir \\
                 --multimer_name=$multimer_name \\
                 --fasta=$align_task_out/${fold_name}.fa
            fi                              
                        
            echo "generating PAE, plDDT plots and JSON files"
            $python_bin -u ${OPENFOLD_HOME}/scripts/generate_pae_plddt_plot.py \\
              --fasta $align_task_out/${fold_name}.fa \\
              --model1_pkl $model_1_pkl \\
              --model2_pkl $model_2_pkl \\
              --model3_pkl $model_3_pkl \\
              --output_dir $__task_output_dir/predictions/ \\
              --basename "${fold_name}" \\
              --interface $__task_output_dir/interfaces.csv
            """
        )()

    for match in dsl.query_all_or_nothing("of-align.*", state="ready"):
        yield dsl.task(
            key=f"of-align-array",
            task_conf=task_conf_func(slurm_options_align),
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()

    for _ in dsl.query_all_or_nothing("of-align-array", state="completed"):
        for match in dsl.query_all_or_nothing("of-fold.*", state="ready"):
            yield dsl.task(
                key=f"of-fold-array",
                task_conf=task_conf_func(slurm_options_fold)
            ).slurm_array_parent(
                children_tasks=match.tasks
            )()



def collabfold_dag(dsl, list_of_multimers, samplesheet, collabfold_task_conf_func):

    colabfold_search_slurm_options = ["--time=24:00:00 --mem=40G --cpus-per-task=8"]

    colabfold_fold_slurm_options = ["--time=8:00:00 --mem=120G --cpus-per-task=12 --gpus-per-node=1"]

    for multimer in list_of_multimers:

        multimer_name = multimer.multimer_name()

        colabfold_search_task = dsl.task(
            key=f"cf-search.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=collabfold_task_conf_func(colabfold_search_slurm_options)
        ).inputs(
            samplesheet=dsl.file(samplesheet),
            multimer_name=multimer_name,
            code_dep1=dsl.file(__file__),
            code_dep2=dsl.file(multimer_code_file()),
            code_dep3=dsl.file(colabfold_analysis.code_path())
        ).outputs(
            fa_out=dsl.file(f'fold.fa'),
            a3m=dsl.file(f'0.a3m')
        ).calls(
            generate_fasta_colabfold
        ).calls("""
            #!/usr/bin/bash

            set -ex
            
            mkdir -p $HOME/.licenses/
            touch $HOME/.licenses/intel

            module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 hh-suite/3.3.0 hmmer/3.2.1 mmseqs2/14-7e284

            TE=$TASK_VENV/bin/activate                      
            echo "will activate env: $TE"
            source $TE

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

    for match in dsl.query_all_or_nothing("cf-search.*", state="ready"):
        yield dsl.task(
            key=f"cf-search-array",
            task_conf=collabfold_task_conf_func(colabfold_search_slurm_options)
        ).slurm_array_parent(
            children_tasks=match.tasks
        )()

    for match in dsl.query_all_or_nothing("cf-search.*", state="completed"):
        for search_task in match.tasks:
            multimer_name = str(search_task.inputs.multimer_name)

            colabfold_batch_task = dsl.task(
                key=f"cf-fold.{multimer_name}",
                is_slurm_array_child=True,
                task_conf=collabfold_task_conf_func(colabfold_fold_slurm_options)
            ).inputs(
                a3m=search_task.outputs.a3m,
                colabfold_analysis_script=dsl.file(colabfold_analysis.code_path()),
                code_dep1=dsl.file(__file__),
                code_dep2=dsl.file(multimer_code_file())
            ).outputs(
                relaxed_pdb=dsl.file(f'fold.fa'),
                unrelaxed_pdb=dsl.file(f'0.a3m'),
                all_results=dsl.file_set("**/*", exclude_pattern="*.pkl|*.pickle"),
            ).calls("""
                #!/usr/bin/bash

                set -e
                
                mkdir -p $HOME/.licenses/
                touch $HOME/.licenses/intel                
                
                module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 hh-suite/3.3.0 hmmer/3.2.1 mmseqs2/14-7e284

                source $TASK_VENV/bin/activate

                export TF_FORCE_UNIFIED_MEMORY="1"
                export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
                export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
                export TF_FORCE_GPU_ALLOW_GROWTH="true"                                                    
                

                echo "running colabfold fold"
                colabfold_batch \\
                  --use-gpu-relax --amber --num-relax 3 \\
                  --num-models 3 \\
                  --num-recycle 30 --recycle-early-stop-tolerance 0.5 \\
                  --model-type auto \\
                  --data $collabfold_db \\
                  $a3m \\
                  $__task_output_dir
                
                search_task_out=$__pipeline_instance_dir/output/cf-search.${multimer_name}

                echo "running AF2multimer-analysis on $__task_output_dir/predictions/"
                mkdir -p $__task_output_dir/predictions/unrelaxed
                mv $__task_output_dir/predictions/*unrelaxed.pdb $__task_output_dir/predictions/unrelaxed/ || true
                $python_bin -u $colabfold_analysis_script \\
                    --pred_folder=$__task_output_dir/predictions \\
                    --out_folder=$__task_output_dir \\
                    --multimer_name=$multimer_name \\
                    --fasta=$search_task_out/fold.fa

                echo "done"

            """)()
            yield colabfold_batch_task

        for match in dsl.query_all_or_nothing("cf-fold.*", state="ready"):
            yield dsl.task(
                key=f"cf-fold-array",
                task_conf=collabfold_task_conf_func(colabfold_fold_slurm_options)
            ).slurm_array_parent(
                children_tasks=match.tasks
            )()


def aggregate_report_task(dsl):

    pipeline_instance_dir_basename = os.path.basename(dsl.pipeline_instance_dir())

    yield dsl.task(
        key=f"of-aggregate-report"
    ).outputs(
        interfaces_csv=dsl.file(f"{pipeline_instance_dir_basename}.interfaces.csv"),
        summary_csv=dsl.file(f"{pipeline_instance_dir_basename}.summary.csv"),
        contacts_csv=dsl.file(f"{pipeline_instance_dir_basename}.contacts.csv")
    ).calls(
        generate_aggregate_report
    )()


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

    for match_completed_openfold_tasks in dsl.query_all_or_nothing("analysis-openfold.*"):
        for match_completed_collabfold_tasks in dsl.query_all_or_nothing("colabfold_batch.*"):
            yield from aggregate_report_task(dsl)





def colabfold_pipeline():

    from dpfold.task_confs import cc_remote_task_conf_func_func

    def p(dsl):

        errors, samplesheet, multimers, pipeline_instance_args = parse_and_validate_input_files(dsl.pipeline_instance_dir())

        tc = cc_remote_task_conf_func_func(pipeline_instance_args)

        yield from collabfold_dag(dsl, multimers, samplesheet, tc)

        for _ in dsl.query_all_or_nothing("colabfold_batch.*"):
            yield from aggregate_report_task(dsl)

    return DryPipe.create_pipeline(p)


def openfold_pipeline():

    from dpfold.task_confs import gh_task_conf

    def p(dsl):
        samplesheet = os.path.join(dsl.pipeline_instance_dir(), "samplesheet.tsv")

        multimers = parse_multimer_list_from_samplesheet(samplesheet)

        yield from openfold_dag(dsl, multimers, samplesheet, gh_task_conf)

        for _ in dsl.query_all_or_nothing("of-fold.*"):
            yield from aggregate_report_task(dsl)

    return DryPipe.create_pipeline(p)


def combined_pipeline():

    from dpfold.task_confs import gh_task_conf, narval_task_conf

    return DryPipe.create_pipeline(
        lambda dsl: combined_pipeline_dag(dsl, gh_task_conf, narval_task_conf)
    )
