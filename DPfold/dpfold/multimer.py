import os
from itertools import islice, groupby
from dataclasses import dataclass
from typing import List


@dataclass
class Protein:
    name: str
    n_occurences: int
    pdb: str
    seq: str


class Multimer:

    def __init__(self, proteins, line_number_in_samplesheet):
        self.proteins = proteins
        self.line_number_in_samplesheet = line_number_in_samplesheet

    def protein_count(self):
        return len(self.proteins)

    def multimer_name(self):
        return "-".join([
            f"{protein.name}_{protein.n_occurences}"
            for protein in self.proteins
        ])

    def __str__(self):
        return self.multimer_name()

    def generate_fasta_colabfold(self, fa_out):

        fa_header = "_".join([
            f"{protein.name}_{protein.n_occurences}"
            for protein in self.proteins
        ])

        fa_seqs = []
        for protein in self.proteins:
            fa_seqs.extend([protein.seq] * protein.n_occurences)

        with open(fa_out, 'w') as f:
            f.write(f">{fa_header}\n")
            f.write(":".join(fa_seqs))
            f.write("\n")

    def sequence_length(self):
        res = 0
        for protein in self.proteins:
            res += len(protein.seq) * protein.n_occurences
        return res

    def generate_openfold_fold_name(self):

        fa_header = []

        for protein in self.proteins:
            for x in range(1, protein.n_occurences + 1):
                fa_header.extend([f"{protein.name}_{x}"])

        return "-".join(fa_header)

    def generate_fasta_openfold(self, fa_out):

        fa_header = []
        fa_seq = []

        for protein in self.proteins:
            for x in range(1, protein.n_occurences + 1):
                fa_header.extend([f"{protein.name}_{x}"])
            fa_seq.extend([protein.seq] * protein.n_occurences)

        with open(fa_out, 'w') as f:
            f.write("\n".join(
                [f">{h}\n{s}" for h, s in zip(fa_header, fa_seq)]
            ))


    def generate_pdb(self, output_dir):

        import requests

        pdb_dir = os.path.join(output_dir, "pdb")
        if not os.path.exists(pdb_dir):
            os.makedirs(pdb_dir)

        for protein in self.proteins:
            pdb_list = protein.pdb.split(",")
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


def parse_multimer_list_from_samplesheet(samplesheet, single_multimer_name=None, include_single_prots=True) -> List[Multimer]:

    def rows():
        with open(samplesheet) as f:
            c = 1
            for line in f.readlines():
                line = line.rstrip()
                if c == 1:
                    pass
                elif line.startswith("#"):
                    pass
                else:
                    yield c, line.split("\t")

                c +=1

    def g():
        for line_number, row in rows():
            def prots_in_row():
                i = iter(row)
                while prot_rows := list(islice(i, 4)):
                    name = prot_rows[0]
                    if "-" in name:
                        raise Exception(f"illegal prot name on line {line_number}, can't use '-' ")
                    n_occurences = int(prot_rows[1])
                    pdb = prot_rows[2]
                    seq = prot_rows[3]
                    yield Protein(name, n_occurences, pdb, seq)

            m = Multimer(list(prots_in_row()), line_number)

            if not include_single_prots:
                if m.protein_count() == 1:
                    print(
                        f"Warning: line {m.line_number_in_samplesheet} ({m.multimer_name()}) excluded, " +
                        "because it contains only one prot"
                    )
                    continue

            if single_multimer_name is None:
                yield m
            else:
                if m.multimer_name() == single_multimer_name:
                    yield m
                    break

    res = list(g())

    def check_duplicates():
        def mn(m):
            return m.multimer_name()
        for n, multimers in groupby(sorted(res, key=mn), key=mn):
            multimers = list(multimers)
            if len(multimers) > 1:
                lines = [str(m.line_number_in_samplesheet) for m in multimers]
                raise Exception(
                    f"duplicate multimer names {n} in {samplesheet}, at lines {','.join(lines)}"
                )

    check_duplicates()

    if single_multimer_name is None:
        return res

    if len(res) == 0:
        raise Exception(f"multimer {single_multimer_name} not found in {samplesheet}")
    elif len(res) > 1:
        lines = [m.line_number_in_samplesheet for m in res]
        raise Exception(f"multiple multimer with name {single_multimer_name} found in {samplesheet}, lines: {lines}")

    return res


def file_path():
    return __file__


