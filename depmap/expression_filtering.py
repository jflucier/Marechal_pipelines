import pandas as pd
import argparse
import sys


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Filter DepMap expression data by quartiles for a specific gene.')

    # Define parameters
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Path to save the output CSV file')
    parser.add_argument('-g', '--gene', default='SLFN11 (91607)', help='The gene column name (default: SLFN11 (91607))')

    args = parser.parse_args()

    try:
        # Load only necessary columns to save memory
        df = pd.read_csv(args.input, usecols=['ModelID', args.gene])

        # Calculate quartiles (25th and 75th percentiles)
        q1 = df[args.gene].quantile(0.25)
        q3 = df[args.gene].quantile(0.75)

        # Filter for low (<= Q1) and high (>= Q3) expression
        low_expr = df[df[args.gene] <= q1].copy()
        high_expr = df[df[args.gene] >= q3].copy()

        # Add quartile labels
        low_expr['quartile'] = 'low'
        high_expr['quartile'] = 'high'

        # Combine and format results
        result = pd.concat([low_expr, high_expr])
        result = result.rename(columns={'ModelID': 'model id', args.gene: 'expression'})

        # Save to file
        result[['model id', 'expression', 'quartile']].to_csv(args.output, index=False)

        print(f"Processing complete.")
        print(f"Output saved to: {args.output}")
        print(f"Thresholds for {args.gene}: Q1={q1:.4f}, Q3={q3:.4f}")

    except FileNotFoundError:
        print(f"Error: The file '{args.input}' was not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Column '{args.gene}' or 'ModelID' not found in the input file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
