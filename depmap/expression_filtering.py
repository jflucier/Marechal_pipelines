import pandas as pd
import argparse
import sys


def find_full_column_name(short_name, columns):
    """Matches a gene symbol (e.g., 'BRCA1') to its DepMap column (e.g., 'BRCA1 (672)')."""
    if short_name in columns:
        return short_name
    matches = [c for c in columns if c.startswith(f"{short_name} (")]
    return matches[0] if matches else None


def get_top_correlations_list(target_gene, crispr_df, model_subset):
    """Calculates Spearman correlation and returns the top 200 rows."""
    subset_df = crispr_df.loc[crispr_df.index.isin(model_subset)]

    if subset_df.empty:
        return None

    # Calculate spearman correlation
    correlations = subset_df.corrwith(subset_df[target_gene], method='spearman')
    correlations = correlations.drop(labels=[target_gene], errors='ignore').dropna()

    corr_df = correlations.reset_index()
    corr_df.columns = ['other gene', 'spearman correlation value']
    corr_df['gene'] = target_gene

    # Get top 200 absolute values
    corr_df['abs_val'] = corr_df['spearman correlation value'].abs()
    top_200 = corr_df.sort_values(by='abs_val', ascending=False).head(200)

    return top_200[['gene', 'other gene', 'spearman correlation value']]


def main():
    parser = argparse.ArgumentParser(description='Correlate CRISPR scores based on expression quartiles.')
    parser.add_argument('-i', '--input', required=True, help='Input expression CSV')
    parser.add_argument('-o', '--output', required=True, help='Output path for expression quartile file')
    parser.add_argument('-g', '--refgene', default='SLFN11 (91607)', help='Reference gene column')
    parser.add_argument('-ddr', '--ddrgenes', required=True, help='DDR gene list text file')
    parser.add_argument('-c', '--crispr_input', required=True, help='CRISPR score CSV')
    args = parser.parse_args()

    try:
        # 1. Load Expression and Categorize
        print(f"--- Categorizing expression quartiles for {args.refgene} ---")
        expr_df = pd.read_csv(args.input, usecols=['ModelID', args.refgene])
        expr_df['quartile'] = pd.qcut(expr_df[args.refgene], q=4, labels=['low', 'low-mid', 'high-mid', 'high'])
        expr_df.rename(columns={'ModelID': 'model id', args.refgene: 'expression'}).to_csv(args.output, index=False)

        # 2. Load CRISPR Data
        print("Loading CRISPR data (indexing ModelID)...")
        crispr_df = pd.read_csv(args.crispr_input, index_col=0)
        all_crispr_cols = crispr_df.columns.tolist()

        # 3. Load and Match DDR Genes
        with open(args.ddrgenes, 'r') as f:
            raw_ddr_list = [line.strip() for line in f if line.strip()]

        ddr_list = []
        for g in raw_ddr_list:
            full_name = find_full_column_name(g, all_crispr_cols)
            if full_name:
                ddr_list.append(full_name)
            else:
                print(f"  Warning: Gene '{g}' not found in CRISPR columns.")

        total_genes = len(ddr_list)
        if total_genes == 0:
            print("Error: No valid DDR genes found to process.")
            return

        # 4. Correlation Loops
        subsets = {
            "all": expr_df['ModelID'],
            "low": expr_df[expr_df['quartile'] == 'low']['ModelID'],
            "high": expr_df[expr_df['quartile'] == 'high']['ModelID']
        }
        results_accum = {"all": [], "low": [], "high": []}

        print(f"Starting correlations for {total_genes} genes...")
        for idx, ddr_gene in enumerate(ddr_list, 1):
            print(f"Processing: {ddr_gene} ({idx}/{total_genes})")

            for key in subsets:
                res = get_top_correlations_list(ddr_gene, crispr_df, subsets[key])
                if res is not None:
                    results_accum[key].append(res)

        # 5. Save Final Consolidated Files
        for key in results_accum:
            if results_accum[key]:
                output_fn = f"spearman.{key}.tsv"
                pd.concat(results_accum[key]).to_csv(output_fn, sep='\t', index=False)
                print(f"Saved: {output_fn}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
