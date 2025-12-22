import pandas as pd
import argparse
import sys
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_full_column_name(short_name, columns):
    """Matches a gene symbol to DepMap formats (Entrez or Ensembl)."""
    if short_name in columns:
        return short_name
    matches = [c for c in columns if c.startswith(f"{short_name} (")]
    return matches[0] if matches else None


def process_single_gene(ddr_gene, crispr_df, model_ids_dict):
    """Worker function: returns results to the main process for collection."""
    local_gene_results = {}
    for key, model_ids in model_ids_dict.items():
        subset_df = crispr_df.loc[crispr_df.index.isin(model_ids)]

        if subset_df.empty or ddr_gene not in subset_df.columns:
            continue

        # Spearman correlation
        correlations = subset_df.corrwith(subset_df[ddr_gene], method='spearman')
        correlations = correlations.drop(labels=[ddr_gene], errors='ignore').dropna()

        corr_df = correlations.reset_index()
        corr_df.columns = ['other gene', 'spearman correlation value']
        corr_df['gene'] = ddr_gene
        corr_df['abs_val'] = corr_df['spearman correlation value'].abs()

        top_200 = corr_df.sort_values(by='abs_val', ascending=False).head(200)
        local_gene_results[key] = top_200[['gene', 'other gene', 'spearman correlation value']]

    return ddr_gene, local_gene_results


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
        os.makedirs(out_dir, exist_ok=True)

    try:
        # 1. Load Expression and Categorize
        header_check = pd.read_csv(args.input, nrows=0)
        id_col_name = header_check.columns[0]  # Get the name of the first column

        # Resolve the reference gene full name in expression file
        expr_gene = find_full_column_name(args.refgene.split(' ')[0], header_check.columns)
        if not expr_gene:
            expr_gene = args.refgene  # Fallback to literal

        print(f"--- Categorizing expression for {expr_gene} (ID col: {id_col_name}) ---")
        expr_df = pd.read_csv(args.input, usecols=[id_col_name, expr_gene])
        expr_df = expr_df.rename(columns={id_col_name: 'ModelID', expr_gene: 'expression_val'})

        # Quartile calculation
        expr_df['quartile'] = pd.qcut(expr_df['expression_val'], q=4, labels=['low', 'low-mid', 'high-mid', 'high'])

        # Save expression categorization
        expr_df.rename(columns={'ModelID': 'model id', 'expression_val': 'expression'}).to_csv(
            f"{args.output}.expression.tsv", sep='\t', index=False
        )

        # 2. Load CRISPR
        print("Loading CRISPR data...")
        crispr_df = pd.read_csv(args.crispr_input, index_col=0)

        # 3. Match DDR Genes
        with open(args.ddrgenes, 'r') as f:
            # Take only the first word (symbol) from each line
            symbols = [line.strip().split(' ')[0] for line in f if line.strip()]

        ddr_list = []
        for s in symbols:
            matched = find_full_column_name(s, crispr_df.columns)
            if matched:
                ddr_list.append(matched)
            else:
                print(f"  Warning: Gene symbol '{s}' not found in CRISPR columns.")

        # 4. Define Subsets
        subsets = {
            "all": expr_df['ModelID'].tolist(),
            "low": expr_df[expr_df['quartile'] == 'low']['ModelID'].tolist(),
            "high": expr_df[expr_df['quartile'] == 'high']['ModelID'].tolist()
        }

        # 5. Parallel Collection
        master_results = {"all": [], "low": [], "high": []}
        total_to_process = len(ddr_list)
        print(f"Running {total_to_process} genes on {args.threads} threads...")

        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            # Submit tasks
            futures = {executor.submit(process_single_gene, gene, crispr_df, subsets): gene for gene in ddr_list}

            for i, future in enumerate(as_completed(futures), 1):
                # FIXED: Correctly unpack the tuple (gene_name, result_dict)
                gene_name, gene_res_dict = future.result()
                print(f"Finished: {gene_name} ({i}/{total_to_process})")

                for key, df in gene_res_dict.items():
                    if df is not None:
                        master_results[key].append(df)

        print(
            f"Concatenating results: All={len(master_results['all'])}, Low={len(master_results['low'])}, High={len(master_results['high'])}")

        # 6. Save Final Combined Files
        for key in ["all", "low", "high"]:
            if master_results[key]:
                final_df = pd.concat(master_results[key])
                output_fn = f"{args.output}.spearman.{key}.tsv"
                final_df.to_csv(output_fn, sep='\t', index=False)
                print(f"Saved: {output_fn}")
            else:
                print(f"No results to save for condition '{key}'.")

    except Exception as e:
        print(f"\nCritical Error Occurred:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
