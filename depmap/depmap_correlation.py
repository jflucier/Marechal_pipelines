import argparse
import os.path
import numpy as np
import pandas as pd
# from nancorrmp.nancorrmp import NaNCorrMp

np.random.seed(0)

def gen_corr_matrix(in_f, out_m):
    print("Reading depmap data")
    df = pd.read_csv(
        in_f,
        index_col=0,
        sep=","
    )

    print("Running correlation")
    df2 = NaNCorrMp.calculate(df, n_jobs=32)

    print("Output correlation")
    df2.to_csv(
        out_m,
        sep="\t"
    )

def extract_top_hits(t, nbr_hits, out_filtered):
    print(f"reading correlation matrix {t}")
    df = pd.read_csv(
        t,
        sep="\t",
        index_col = 0
    )

    # RFWD3 (55159)
    top_hits = pd.DataFrame()
    for column in df:
        print(f"running {column}")
        sorted = df.sort_values(column)
        filtered_data = sorted.drop([column] , axis=0)
        d = filtered_data[column]
        # d['gene'] = column
        r = pd.DataFrame({'gene': column, 'target': d.index, 'corr': d.values})
        top_hits_highcorr = r.tail(nbr_hits)
        top_hits_lowcorr = r.head(nbr_hits)
        top_hits = pd.concat([top_hits, top_hits_lowcorr, top_hits_highcorr])

    top_hits.to_csv(
        out_filtered,
        sep="\t",
        index = False
    )


# def main():
#     # in_f = "/home/def-marechal/analysis/20250127_ligasee3_depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv"
#     # out_m = "/home/def-marechal/analysis/20250127_ligasee3_depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.correlation.tsv",
#     # gen_corr_matrix(in_f, out_m)
#
#     t = "/storage/Documents/service/biologie/marechal/analysis/20250127_ligasee3_depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.correlation.test.tsv"
#     out_filtered = "/storage/Documents/service/biologie/marechal/analysis/20250127_ligasee3_depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.correlation.test.top10.tsv"
#     nbr_hits = 5
#     extract_top_hits(t, nbr_hits, out_filtered)
#
#     print("done")

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    # mandatory
    argParser.add_argument(
        "-i",
        "--depmap_data",
        help="Depmap",
        required=True
    )

    argParser.add_argument(
        "-o",
        "--corr_matrix",
        help="output correlation matrix file",
        required=True
    )

    argParser.add_argument(
        "-t",
        "--top_hits",
        help="top hit number to return",
        required=False,
        type=int,
        default=250
    )

    args = argParser.parse_args()

    i = args.depmap_data
    o = args.corr_matrix
    t = args.top_hits

    # gen_corr_matrix(
    #     i,
    #     o
    # )

    # i = "/storage/Documents/service/biologie/marechal/analysis/20250127_ligasee3_depmap/CRISPRGeneDependency.corr.test.tsv"
    # o = "/storage/Documents/service/biologie/marechal/analysis/20250127_ligasee3_depmap/CRISPRGeneDependency.corr.test.top5.tsv"
    extract_top_hits(i, t, o)