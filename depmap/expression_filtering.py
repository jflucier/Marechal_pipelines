import pandas as pd
import argparse
import sys


def get_top_correlations_list(target_gene, crispr_df, model_subset):
    """Calculates Spearman correlation and returns the top 200 rows as a DataFrame."""
    if target_gene not in crispr_df.columns:
        print(f"  Warning: {target_gene} not found in CRISPR data. Skipping...")
        return None

    # Filter CRISPR data for specific models
    subset_df = crispr_df.loc[crispr_df.index.isin(model_subset)]

    # Calculate spearman correlation of target_gene against all columns
    # We use .corrwith for efficiency when comparing one column to many
    correlations = subset_df.corrwith(subset_df[target_gene], method='spearman')

    # Drop the target gene itself
    correlations = correlations.drop(labels=[target_gene])

    # Create results dataframe
    corr_df = correlations.reset_index()
    corr_df.columns = ['other gene', 'spearman correlation value']
    corr_df['gene'] = target_gene

    # Get top 200 absolute values
    corr_df['abs_val'] = corr_df['spearman correlation value'].abs()
    top_200 = corr_df.sort_values(by='abs_val', ascending=False).head(200)

    return top_200[['gene', 'other gene', 'spearman correlation value']]


def main():
    parser = argparse.ArgumentParser(description='Correlate CRISPR scores based on expression quartiles.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV expression file')
    parser.add_argument('-o', '--output', required=True, help='Path for expression quartile output CSV')
    parser.add_argument('-g', '--refgene', default='SLFN11 (91607)',
                        help='Reference gene column name (e.g. SLFN11 (91607))')
    parser.add_argument('-ddr', '--ddrgenes', required=True, help='File with DDR gene names (one per line)')
    parser.add_argument('-c', '--crispr_input', required=True, help='Path to CRISPR score CSV')
    args = parser.parse_args()

    try:
        # 1. Load Expression Data and Categorize
        print("Categorizing expression quartiles...")
        expr_df = pd.read_csv(args.input, usecols=['ModelID', args.refgene])
        labels = ['low', 'low-mid', 'high-mid', 'high']
        expr_df['quartile'] = pd.qcut(expr_df[args.refgene], q=4, labels=labels)

        # Save quartile mapping
        expr_df.rename(columns={'ModelID': 'model id', args.refgene: 'expression'}).to_csv(args.output, index=False)

        # 2. Load CRISPR Data
        print("Loading CRISPR data (this may take a moment)...")
        # index_col=0 assumes ModelID is the first column
        crispr_df = pd.read_csv(args.crispr_input, index_col=0)

        # 3. Load DDR Genes list
        with open(args.ddrgenes, 'r') as f:
            ddr_list = [line.strip() for line in f if line.strip()]

        # Define model subsets
        subsets = {
            "all": expr_df['ModelID'],
            "low": expr_df[expr_df['quartile'] == 'low']['ModelID'],
            "high": expr_df[expr_df['quartile'] == 'high']['ModelID']
        }

        # Accumulators for the final files
        results_accum = {"all": [], "low": [], "high": []}

        # 4. Correlation Loops
        for ddr_gene in ddr_list:
            print(f"Processing: {ddr_gene}")

            for key in subsets:
                res = get_top_correlations_list(ddr_gene, crispr_df, subsets[key])
                if res is not None:
                    results_accum[key].append(res)

        # 5. Save consolidated files
        for key in results_accum:
            if results_accum[key]:
                final_df = pd.concat(results_accum[key])
                output_fn = f"spearman.{key}.tsv"
                final_df.to_csv(output_fn, sep='\t', index=False)
                print(f"Saved consolidated file: {output_fn}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Required column not found. Check if {args.refgene} and ModelID exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
