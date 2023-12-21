import argparse
import os
import sys
import pandas as pd
import json


def generate_plddt_graph(in_f, out):
    # removing the new line characters
    df = pd.DataFrame()
    with open(in_f) as f:
        for json_p in f:
            file_name = os.path.basename(json_p.strip())
            print("extracting plddt from {}".format(file_name))

            with open(json_p.strip()) as json_f:
                data = json.load(json_f)
                plddt = data['plddt'][0:281]
                df[file_name] = plddt

    df.to_csv(
        out,
        sep='\t'
    )

if __name__ == '__main__':
    # argParser = argparse.ArgumentParser()
    #
    # # mandatory
    # argParser.add_argument(
    #     "-i",
    #     "--input",
    #     help="path to file with list of colabfold models json files",
    #     required=True
    # )
    #
    # argParser.add_argument(
    #     "-o",
    #     "--out",
    #     help="path to output tsv file",
    #     required=True
    # )
    #
    # args = argParser.parse_args()
    # input = args.input
    # out = args.out

    in_f = "/storage/Documents/service/biologie/marechal/analysis/20230922_fold/plDDT_graphs/batch8_json.list"
    out = "/storage/Documents/service/biologie/marechal/analysis/20230922_fold/plDDT_graphs/batch8_json.list.tsv"

    generate_plddt_graph(
        in_f,
        out
    )
