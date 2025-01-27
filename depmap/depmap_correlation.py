import os.path
import pandas as pd


print("Reading depmap data")
df = pd.read_csv(
    "/home/def-marechal/analysis/20250127_ligasee3_depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv",
    index_col=0,
    sep=","
)

print("Running correlation")
df2=df.corr()

print("Output correlation")
df2.to_csv(
    "/home/def-marechal/analysis/20250127_ligasee3_depmap/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.correlation.tsv",
    sep="\t"
)

print("done")
