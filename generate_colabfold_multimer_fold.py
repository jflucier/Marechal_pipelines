import argparse
import os
import sys

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
        else:
            print(f"Multimer name duplicated: {index}. They must be unique. Please modify your input TSV: {foldsheet} ")
            sys.exit(0)

        print(f"Generating fold job {index} script in {workdir}")
        # generate fasta
        fasta_out = os.path.join(workdir, f"{index}.fa")
        print(f"generating fasta: {fasta_out}")
        generate_fasta(fasta_out, row)

        # if pdb is provided, generate pdb dir struct and download
        generate_pdb(workdir, row)

        # gen submit script:
        script_path = os.path.dirname(__file__)
        tmpl_data = {
            'job_name': f"{index}",
            'colabfold_db': '/home/jflucier/projects/def-marechal/colabfold_db',
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

def generate_pdb(wd,row):
    pdb_dir = os.path.join(wd, "pdb")
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    prot_nbr = 1
    while f"protein{prot_nbr}_PDB" in row.index:
        if not pd.isna(row[f"protein{prot_nbr}_PDB"]):
            pdb_str = row[f"protein{prot_nbr}_PDB"]
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

        prot_nbr = prot_nbr + 1

def generate_fasta(fa_out,row):
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
