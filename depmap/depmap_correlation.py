import os.path
import pandas as pd


def gen_corr_matrix(in_f, out_m):
    print("Reading depmap data")
    df = pd.read_csv(
        in_f,
        index_col=0,
        sep=","
    )

    print("Running correlation")
    df2=df.corr()

    print("Output correlation")
    df2.to_csv(
        out_m,
        sep="\t"
    )

def extract_top_hits(t, nbr_hits, out_filtered):
    df = pd.read_csv(
        t,
        sep="\t",
        index_col = 0
    )

    # RFWD3 (55159)
    top_hits = pd.DataFrame()
    for column in df:
        sorted = df.sort_values(column)
        filtered_data = sorted.drop([column] , axis=0)
        d = filtered_data[column]
        top_hits_highcorr = filtered_data.tail(nbr_hits)
        top_hits_lowcorr = filtered_data.head(nbr_hits)
        top_hits = top_hits.concat([top_hits_highcorr, top_hits_lowcorr])

    top_hits.to_csv(
        out_filtered,
        sep="\t"
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

    args = argParser.parse_args()

    gen_corr_matrix(
        args.depmap_data,
        args.corr_matrix
    )