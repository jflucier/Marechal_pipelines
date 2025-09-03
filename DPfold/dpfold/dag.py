import glob
import os.path
import zipfile
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

        #if "cc_project" not in pipeline_instance_args:
        #    yield "MISSING_ARG", "'cc_project' must be specified in Pipeline Args"

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
def download_pdbs(samplesheet, pdb_folder):

    multimer_batch = parse_multimer_list_from_samplesheet(samplesheet)

    pdb_dir = Path(pdb_folder)
    if not pdb_dir.exists():
        pdb_dir.mkdir(parents=False)

    multimer_batch.download_pdbs(pdb_dir)



@DryPipe.python_call()
def generate_aggregate_report(__pipeline_instance_dir, interfaces_csv, summary_csv, contacts_csv, all_zip, __task_output_dir):

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

    fold_outfile = Path(__pipeline_instance_dir, "output").glob("cf-fold.*/*")

    zip_root = Path(__pipeline_instance_dir, "output")

    excluded_files = [
        "0_predicted_aligned_error_v1.json",
        "fake_home"
    ]

    with zipfile.ZipFile(all_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fof in fold_outfile:
            if fof.name.endswith(".done.txt") or fof.name in excluded_files:
                continue
            zipf.write(fof, arcname=fof.relative_to(zip_root))


        zipf.write(interfaces_csv, arcname="interfaces.csv")
        zipf.write(summary_csv, arcname="summary.csv")
        zipf.write(contacts_csv, arcname="contacts.csv")



def collabfold_dag(dsl, multimer_batch, samplesheet, collabfold_task_conf_func):


    colabfold_search_slurm_options = ["--time=24:00:00 --mem=40G --cpus-per-task=8"]

    colabfold_fold_slurm_options = ["--time=12:00:00 --mem=120G --cpus-per-task=12 --gpus-per-node=1"]

    download_pdbs_task = dsl.task(
        key=f"cf-download-pdbs"
    ).inputs(
        samplesheet=dsl.file(samplesheet)
    ).outputs(
        pdbs=dsl.file("pdbs")
    ).calls(download_pdbs)()

    yield download_pdbs_task

    for multimer in multimer_batch:

        multimer_name = multimer.multimer_name()

        colabfold_search_task = dsl.task(
            key=f"cf-search.{multimer_name}",
            is_slurm_array_child=True,
            task_conf=collabfold_task_conf_func(colabfold_search_slurm_options)
        ).inputs(
            samplesheet=dsl.file(samplesheet),
            multimer_name=multimer_name,
            pdbs=download_pdbs_task.outputs.pdbs,
            fold_name=str(multimer.fold_name()),
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
                code_dep2=dsl.file(multimer_code_file()),
                pdbs=download_pdbs_task.outputs.pdbs,
                multimer_name=multimer_name,
                fold_name=search_task.inputs.fold_name,
                has_pdbs=str("True" if multimer_batch.multimer_by_name(multimer_name).has_pdbs() else "False")
            ).outputs(
                relaxed_pdb=dsl.file(f'fold.fa'),
                unrelaxed_pdb=dsl.file(f'0.a3m'),
                all_results=dsl.file_set("**/*", exclude_pattern="*.pkl|*.pickle"),
            ).calls("""
                #!/usr/bin/bash

                set -ex
                
                mkdir -p $HOME/.licenses/
                touch $HOME/.licenses/intel                
                
                module load StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 openmm/8.0.0 hh-suite/3.3.0 hmmer/3.2.1 mmseqs2/14-7e284

                source $TASK_VENV/bin/activate

                export TF_FORCE_UNIFIED_MEMORY="1"
                export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
                export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
                export TF_FORCE_GPU_ALLOW_GROWTH="true"
                                
                if [[ "$has_pdbs" == "True" ]]; then
                   template_args="--templates 1 --custom-template-path $__pipeline_instance_dir/pdbs"
                else
                   template_args=""
                fi
                
                echo "pb1: $python_bin"
                echo "pb2: $TASK_VENV/bin/python3"

                echo "running colabfold fold"
                colabfold_batch $template_args \\
                  --use-gpu-relax --amber --num-relax 3 \\
                  --num-models 3 \\
                  --num-recycle 30 --recycle-early-stop-tolerance 0.5 \\
                  --model-type auto \\                                    
                  --data $collabfold_db \\
                  $a3m \\
                  $__task_output_dir
                
                search_task_out=$__pipeline_instance_dir/output/cf-search.${multimer_name}

                echo "running AF2multimer-analysis on $__task_output_dir"                                                                
                
                python3 -u $colabfold_analysis_script \\
                    --pred_folder=$__task_output_dir \\
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
        contacts_csv=dsl.file(f"{pipeline_instance_dir_basename}.contacts.csv"),
        all_zip=dsl.file(f"{pipeline_instance_dir_basename}.all.zip"),
    ).calls(
        generate_aggregate_report
    )()


def combined_pipeline_dag(dsl, openfold_task_conf_func, collabfold_task_conf_func):
    from dpfold.openfold_dag import openfold_dag

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

        for _ in dsl.query_all_or_nothing("cf-fold.*"):
            yield from aggregate_report_task(dsl)

    return DryPipe.create_pipeline(p)


def openfold_pipeline():

    from dpfold.task_confs import gh_task_conf
    from dpfold.openfold_dag import openfold_dag

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
