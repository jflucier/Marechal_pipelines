import pandas as pd
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_full_column_name(short_name, columns):
    """Matches a gene symbol (e.g., 'BRCA1') to its DepMap column (e.g., 'BRCA1 (672)')."""
    if short_name in columns:
        return short_name
    matches = [c for c in columns if c.startswith(f"{short_name} (")]
    return matches[0] if matches else None


def process_single_gene(ddr_gene, crispr_df, subsets):
    """Worker function to calculate correlations for one gene across all subsets."""
    gene_results = {"all": None, "low": None, "high": None}

    for key, model_ids in subsets.items():
        subset_df = crispr_df.loc[crispr_df.index.isin(model_ids)]
        if subset_df.empty:
            continue

        # Calculate spearman correlation
        correlations = subset_df.corrwith(subset_df[ddr_gene], method='spearman')
        correlations = correlations.drop(labels=[ddr_gene], errors='ignore').dropna()

        corr_df = correlations.reset_index()
        corr_df.columns = ['other gene', 'spearman correlation value']
        corr_df['gene'] = ddr_gene

        # Get top 200 absolute values
        corr_df['abs_val'] = corr_df['spearman correlation value'].abs()
        top_200 = corr_df.sort_values(by='abs_val', ascending=False).head(200)
        gene_results[key] = top_200[['gene', 'other gene', 'spearman correlation value']]

    return ddr_gene, gene_results


def main():
    parser = argparse.ArgumentParser(description='Parallelized CRISPR correlation by expression quartiles.')
    parser.add_argument('-i', '--input', required=True, help='Input expression CSV')
    parser.add_argument('-o', '--output', required=True, help='Output path for expression quartile file')
    parser.add_argument('-g', '--refgene', default='SLFN11 (91607)', help='Reference gene column')
    parser.add_argument('-ddr', '--ddrgenes', required=True, help='DDR gene list text file')
    parser.add_argument('-c', '--crispr_input', required=True, help='CRISPR score CSV')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of worker processes (default: 4)')
    args = parser.parse_args()

    try:
        # 1. Categorize Expression
        print(f"--- Categorizing expression for {args.refgene} ---")
        expr_df = pd.read_csv(args.input, usecols=['ModelID', args.refgene])
        expr_df['quartile'] = pd.qcut(expr_df[args.refgene], q=4, labels=['low', 'low-mid', 'high-mid', 'high'])
        expr_df.rename(columns={'ModelID': 'model id', args.refgene: 'expression'}).to_csv(args.output, index=False)

        # 2. Load CRISPR Data
        print("Loading CRISPR data...")
        crispr_df = pd.read_csv(args.crispr_input, index_col=0)

        # 3. Match DDR Genes
        with open(args.ddrgenes, 'r') as f:
            raw_list = [line.strip() for line in f if line.strip()]

        ddr_list = [find_full_column_name(g, crispr_df.columns) for g in raw_list]
        ddr_list = [g for g in ddr_list if g is not None]
        total_genes = len(ddr_list)

        # 4. Define Subsets
        subsets = {
            "all": expr_df['ModelID'].tolist(),
            "low": expr_df[expr_df['quartile'] == 'low']['ModelID'].tolist(),
            "high": expr_df[expr_df['quartile'] == 'high']['ModelID'].tolist()
        }

        # 5. Parallel Processing
        results_accum = {"all": [], "low": [], "high": []}
        print(f"Starting parallel processing with {args.threads} workers for {total_genes} genes...")

        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            # Map the worker function to ddr_list
            futures = {executor.submit(process_single_gene, gene, crispr_df, subsets): gene for gene in ddr_list}

            for i, future in enumerate(as_completed(futures), 1):
                gene_name, gene_results = future.result()
                print(f"Finished: {gene_name} ({i}/{total_genes})")

                for key in results_accum:
                    if gene_results[key] is not None:
                        results_accum[key].append(gene_results[key])

        # 6. Save Final Files
        for key in results_accum:
            if results_accum[key]:
                output_fn = f"spearman.{key}.tsv"
                pd.concat(results_accum[key]).to_csv(output_fn, sep='\t', index=False)
                print(f"Final file saved: {output_fn}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
