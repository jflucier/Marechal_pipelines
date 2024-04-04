import argparse
import os
import sys

import pandas as pd
import requests
from string import Template

AF_VERSION = "2.3.2"
ENV = f"/home/jflucier/projects/def-marechal/programs/colabfold_af{AF_VERSION}_env/bin/activate"

def setup_fold(foldsheet, output_dir, account, db):
    folds = pd.read_csv(foldsheet, sep='\t', index_col="multimer_name")

    # cleanup submission script if exists
    if os.path.exists(os.path.join(output_dir, "01_submit_all_colab_search_jobs.sh")):
        os.remove(os.path.join(output_dir, "01_submit_all_colab_search_jobs.sh"))

    if os.path.exists(os.path.join(output_dir, "02_submit_all_colab_fold_jobs.sh")):
        os.remove(os.path.join(output_dir, "02_submit_all_colab_fold_jobs.sh"))

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
        generate_search_script(output_dir, index, db, workdir, fasta_out, account)
        generate_fold_script(output_dir, index, db, workdir, account)

    print(f"\nPlease submit jobs in 2 steps:")
    print(f"Step 1: sh {output_dir}/01_submit_all_colab_search_jobs.sh")
    print(f"Step 2: sh {output_dir}/02_submit_all_colab_fold_jobs.sh")
    print(f"\nMake sure Step 1 completes successfully before running Step2.")

def generate_fold_script(output_dir, index, db, workdir, account):
    script_path = os.path.dirname(__file__)
    fold_tmpl_data = {
        'ENV': ENV,
        'job_name': f"{index}",
        'colabfold_db': db,
        'outdir': f"{workdir}",
        'align_a3m_file': f"{workdir}/0.a3m",
        'script_path': f"{script_path}",
        'account': account
    }

    print(f"Generating colab fold submission script: {workdir}/submit_colab_fold.{index}.sh\n")
    with open(os.path.join(script_path, "submit_colabfold.fold.tmpl"), 'r') as f:
        src = Template(f.read())
        result = src.substitute(fold_tmpl_data)
        with open(os.path.join(workdir, f"submit_colab_fold.{index}.sh"), 'w') as out:
            out.write(result)

    with open(os.path.join(output_dir, "02_submit_all_colab_fold_jobs.sh"), 'a') as o:
        o.write(f"sbatch {workdir}/submit_colab_fold.{index}.sh\n")


def generate_search_script(output_dir, index, db, workdir, fasta_out, account):
    script_path = os.path.dirname(__file__)
    search_tmpl_data = {
        'ENV': ENV,
        'job_name': f"{index}",
        'colabfold_db': db,
        'outdir': f"{workdir}",
        'fasta': f"{fasta_out}",
        'account': account
    }

    print(f"Generating colab search submission script: {workdir}/submit_colab_search.{index}.sh\n")
    with open(os.path.join(script_path, "submit_colabfold.search.tmpl"), 'r') as f:
        src = Template(f.read())
        result = src.substitute(search_tmpl_data)
        with open(os.path.join(workdir, f"submit_colab_search.{index}.sh"), 'w') as out:
            out.write(result)

    with open(os.path.join(output_dir, "01_submit_all_colab_search_jobs.sh"), 'a') as o:
        o.write(f"sbatch {workdir}/submit_colab_search.{index}.sh\n")


def generate_pdb(wd, row):
    pdb_dir = os.path.join(wd, "pdb")
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


def generate_fasta(fa_out, row):
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

    argParser.add_argument(
        "-a",
        "--account",
        help="your allocation account",
        type=str,
        default="def-marechal"
    )

    argParser.add_argument(
        "-db",
        "--database",
        help="your colabfold database path",
        type=str,
        default="/home/jflucier/projects/def-marechal/programs/colabfold_db"
    )

    args = argParser.parse_args()

    setup_fold(
        args.foldsheet,
        args.output,
        args.account,
        args.database,
    )
