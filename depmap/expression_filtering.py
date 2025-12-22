import pandas as pd
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_full_column_name(short_name, columns):
    """
    Matches a gene symbol (e.g., 'BRCA1') to various DepMap column formats
    (e.g., 'BRCA1 (672)' or 'BRCA1 (ENSG...)').
    """
    if short_name in columns:
        return short_name

    # Try finding symbol + Entrez ID (91607) format
    matches_entrez = [c for c in columns if c.startswith(f"{short_name} (") and not '(ENSG' in c]
    if matches_entrez:
        return matches_entrez[0]

    # Try finding symbol + Ensembl ID (ENSG...) format
    matches_ensg = [c for c in columns if c.startswith(f"{short_name} (ENSG")]
    if matches_ensg:
        return matches_ensg[0]

    return None


def process_single_gene(ddr_gene, crispr_df, subsets):
    """Worker function to calculate correlations."""
    gene_results = {"all": None, "low": None, "high": None}

    for key, model_ids in subsets.items():
        subset_df = crispr_df.loc[crispr_df.index.isin(model_ids)]
        if subset_df.empty:
            continue

        correlations = subset_df.corrwith(subset_df[ddr_gene], method='spearman')
        correlations = correlations.drop(labels=[ddr_gene], errors='ignore').dropna()

        corr_df = correlations.reset_index()
        corr_df.columns = ['other gene', 'spearman correlation value']
        corr_df['gene'] = ddr_gene

        corr_df['abs_val'] = corr_df['spearman correlation value'].abs()
        top_200 = corr_df.sort_values(by='abs_val', ascending=False).head(200)
        gene_results[key] = top_200[['gene', 'other gene', 'spearman correlation value']]

    return ddr_gene, gene_results


def main():
    parser = argparse.ArgumentParser(description='Parallelized CRISPR correlation by expression quartiles.')
    parser.add_argument('-i', '--input', required=True, help='Input expression CSV')
    parser.add_argument('-o', '--output', required=True, help='Output prefix for files (e.g., correlation/24Q4)')
    parser.add_argument('-g', '--refgene', default='SLFN11', help='Reference gene symbol or name (e.g. SLFN11 (91607))')
    parser.add_argument('-ddr', '--ddrgenes', required=True, help='DDR gene list text file')
    parser.add_argument('-c', '--crispr_input', required=True, help='CRISPR score CSV')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of worker processes')
    args = parser.parse_args()

    try:
        # 1. Flexible Header Detection for Expression File
        header_check = pd.read_csv(args.input, nrows=0)
        all_expr_cols = header_check.columns.tolist()
        id_col_name = all_expr_cols[0]  # Assume the very first column is the ID

        # Verify the reference gene exists
        ref_gene_name = find_full_column_name(args.refgene.split(' ')[0], all_expr_cols)
        if not ref_gene_name:
            raise KeyError(f"Reference gene '{args.refgene}' not found in expression file headers.")

        print(f"--- Categorizing expression for {ref_gene_name} (ID col: {id_col_name}) ---")

        # Load data using identified column names
        expr_df = pd.read_csv(args.input, usecols=[id_col_name, ref_gene_name])
        expr_df = expr_df.rename(columns={id_col_name: 'ModelID', ref_gene_name: 'expression_val'})

        expr_df['quartile'] = pd.qcut(expr_df['expression_val'], q=4, labels=['low', 'low-mid', 'high-mid', 'high'])

        output_fn_expr = f"{args.output}.expression.tsv"
        expr_df.rename(columns={'ModelID': 'model id', 'expression_val': 'expression'}).to_csv(output_fn_expr,
                                                                                               index=False, sep='\t')

        # 2. Load CRISPR Data
        print("Loading CRISPR data...")
        crispr_df = pd.read_csv(args.crispr_input, index_col=0)
        all_crispr_cols = crispr_df.columns.tolist()

        # 3. Load and Match DDR Genes
        with open(args.ddrgenes, 'r') as f:
            raw_list = [line.strip().split(' ')[0] for line in f if line.strip()]  # Extract just the symbol

        ddr_list = []
        for g_symbol in raw_list:
            full_name = find_full_column_name(g_symbol, all_crispr_cols)
            if full_name:
                ddr_list.append(full_name)
            else:
                print(f"  Warning: Gene '{g_symbol}' not found in CRISPR columns.")

        total_genes = len(ddr_list)
        if total_genes == 0:
            print("Error: No valid DDR genes found in the CRISPR dataset.")
            sys.exit(1)

        # ... (Steps 4, 5, 6 remain the same as previous script) ...
        # Define Subsets
        subsets = {
            "all": expr_df['ModelID'].tolist(),
            "low": expr_df[expr_df['quartile'] == 'low']['ModelID'].tolist(),
            "high": expr_df[expr_df['quartile'] == 'high']['ModelID'].tolist()
        }

        # Parallel Processing
        results_accum = {"all": [], "low": [], "high": []}
        print(f"Starting parallel processing with {args.threads} workers for {total_genes} genes...")

        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(process_single_gene, gene, crispr_df, subsets): gene for gene in ddr_list}

            for i, future in enumerate(as_completed(futures), 1):
                gene_name, gene_results = future.result()
                print(f"Finished: {gene_name} ({i}/{total_genes})")

                for key in results_accum:
                    if gene_results[key] is not None:
                        results_accum[key].append(gene_results[key])

        # Save Final Files
        for key in results_accum:
            if results_accum[key]:
                output_fn = f"{args.output}.spearman.{key}.tsv"
                pd.concat(results_accum[key]).to_csv(output_fn, sep='\t', index=False)
                print(f"Final file saved: {output_fn}")


    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
