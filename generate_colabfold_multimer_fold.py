import argparse
import os

import pandas as pd
import requests
from string import Template

def setup_multimer_fold(foldsheet,output_dir):
    folds = pd.read_csv(foldsheet, sep='\t', index_col="multimer_name")

    # cleanup submission script if exists
    if os.path.exists(os.path.join(output_dir, "submit_all_multimer_jobs.sh")):
        os.remove(os.path.join(output_dir, "submit_all_multimer_jobs.sh"))

    for index, row in folds.iterrows():
        # Path
        workdir = os.path.join(output_dir, index)
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        print(f"Generating fold job {index} script in {workdir}")
        # generate fasta
        fasta_out = os.path.join(workdir, f"{index}.fa")
        generate_fasta(
            fasta_out,
            row["protein1_name"],
            row["protein1_nbr"],
            row["protein1_seq"],
            row["protein2_name"],
            row["protein2_nbr"],
            row["protein2_seq"],
        )

        # if pdb is provided, generate pdb dir struct and download
        if row["protein1_PDB"]:
            generate_pdb(workdir,row["protein1_PDB"])

        if row["protein2_PDB"]:
            generate_pdb(workdir,row["protein2_PDB"])

        # gen submit script:
        script_path = os.path.dirname(__file__)
        tmpl_data = {
            'job_name': f"{index}",
            'colabfold_db': '/home/jflucier/scratch/colabfold_db',
            'outdir': f"{workdir}",
            'fasta': f"{fasta_out}",
            'out_analysis': f"{workdir}_analysis",
            'script_path': f"{script_path}"
        }

        print(f"Generating submission script: {workdir}/submit_colabfold_multimer.{index}.sh\n")
        with open(os.path.join(script_path, "submit_colabfold_multimer.tmpl"), 'r') as f:
            src = Template(f.read())
            result = src.substitute(tmpl_data)
            with open(os.path.join(workdir, f"submit_colabfold_multimer.{index}.sh"), 'w') as out:
                out.write(result)

        with open(os.path.join(output_dir, "submit_all_multimer_jobs.sh"), 'a') as o:
            o.write(f"sbatch {workdir}/submit_colabfold_multimer.{index}.sh\n")

    print(f"\nTo submit, please run: sh {output_dir}/submit_all_multimer_jobs.sh\n")

def generate_pdb(wd,pdb_str):
    pdb_dir = os.path.join(wd, "pdb")
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    pdb_list=pdb_str.split(",")
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




def generate_fasta(fa_out,p1_name,p1_nbr,p1_seq,p2_name,p2_nbr,p2_seq):
    with open(fa_out, 'w') as f:
        f.write(f">{p1_name}_{p1_nbr}_{p2_name}_{p2_nbr}\n")
        seq_items = [p1_seq] * p1_nbr
        seq_items.extend([p2_seq] * p2_nbr)
        seq = ":".join(seq_items)
        f.write(f"{seq}\n")

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    # mandatory
    argParser.add_argument(
        "-fs",
        "--foldsheet",
        help="your tab seperated fold sheet: "
             "multimer_name<tab>protein1_name<tab>protein1_nbr<tab"
             ">protein1_PDB<tab>protein1_seq<tab>protien2_name<tab"
             ">protein2_nbr<tab>protein2_PDB<tab>protein2_seq",
        required=True
    )
    argParser.add_argument(
        "-o",
        "--output",
        help="your output directory",
        required=True
    )

    args = argParser.parse_args()

    setup_multimer_fold(
        args.foldsheet,
        args.output
    )
