import pandas as pd
import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_full_column_name(short_name, columns):
    """Matches a gene symbol to DepMap formats (Entrez or Ensembl)."""
    if short_name in columns:
        return short_name
    # Search for symbol at start of string followed by a space and parenthesis
    matches = [c for c in columns if c.startswith(f"{short_name} (")]
    return matches[0] if matches else None


def process_single_gene(ddr_gene, crispr_df, subsets):
    """Worker function: returns results to the main process for collection."""
    local_gene_results = {}
    for key, model_ids in subsets.items():
        subset_df = crispr_df.loc[crispr_df.index.isin(model_ids)]
        if subset_df.empty or ddr_gene not in subset_df.columns:
            continue

        # Spearman correlation against all columns
        correlations = subset_df.corrwith(subset_df[ddr_gene], method='spearman')
        correlations = correlations.drop(labels=[ddr_gene], errors='ignore').dropna()

        corr_df = correlations.reset_index()
        corr_df.columns = ['other gene', 'spearman correlation value']
        corr_df['gene'] = ddr_gene
        corr_df['abs_val'] = corr_df['spearman correlation value'].abs()

        top_200 = corr_df.sort_values(by='abs_val', ascending=False).head(200)
        local_gene_results[key] = top_200[['gene', 'other gene', 'spearman correlation value']]

    return local_gene_results


def main():
    parser = argparse.ArgumentParser(description='Parallelized CRISPR correlation.')
    parser.add_argument('-i', '--input', required=True, help='Input expression CSV')
    parser.add_argument('-o', '--output', required=True, help='Output prefix (e.g., correlation/24Q4)')
    parser.add_argument('-g', '--refgene', default='SLFN11', help='Reference gene symbol')
    parser.add_argument('-ddr', '--ddrgenes', required=True, help='DDR gene list file')
    parser.add_argument('-c', '--crispr_input', required=True, help='CRISPR score CSV')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads')
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    try:
        # 1. Load Expression and Categorize
        header_check = pd.read_csv(args.input, nrows=0)
        id_col = header_check.columns[0]
        expr_gene = find_full_column_name(args.refgene.split(' ')[0], header_check.columns)

        print(f"--- Categorizing expression for {expr_gene} ---")
        expr_df = pd.read_csv(args.input, usecols=[id_col, expr_gene])
        expr_df['quartile'] = pd.qcut(expr_df[expr_gene], q=4, labels=['low', 'low-mid', 'high-mid', 'high'])
        expr_df.to_csv(f"{args.output}.expression.tsv", sep='\t', index=False)

        # 2. Load CRISPR
        print("Loading CRISPR data...")
        crispr_df = pd.read_csv(args.crispr_input, index_col=0)

        # 3. Match DDR Genes
        with open(args.ddrgenes, 'r') as f:
            symbols = [line.strip().split(' ')[0] for line in f if line.strip()]

        ddr_list = [find_full_column_name(s, crispr_df.columns) for s in symbols]
        ddr_list = [g for g in ddr_list if g]

        subsets = {
            "all": expr_df[id_col].tolist(),
            "low": expr_df[expr_df['quartile'] == 'low'][id_col].tolist(),
            "high": expr_df[expr_df['quartile'] == 'high'][id_col].tolist()
        }

        # 4. Parallel Collection
        master_results = {"all": [], "low": [], "high": []}
        print(f"Running {len(ddr_list)} genes on {args.threads} threads...")

        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            # Submit tasks
            futures = {executor.submit(process_single_gene, gene, crispr_df, subsets): gene for gene in ddr_list}

            for i, future in enumerate(as_completed(futures), 1):
                gene_res_dict = future.result()  # This brings the data back to main process
                print(f"Finished: {futures[future]} ({i}/{len(ddr_list)})")

                for key, df in gene_res_dict.items():
                    master_results[key].append(df)

        # 5. Save Final Combined Files
        for key in ["all", "low", "high"]:
            if master_results[key]:
                final_df = pd.concat(master_results[key])
                final_df.to_csv(f"{args.output}.spearman.{key}.tsv", sep='\t', index=False)
                print(f"Saved: {args.output}.spearman.{key}.tsv")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
