import pandas as pd
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Categorize all models by gene expression quartiles.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Path for output CSV')
    parser.add_argument('-g', '--gene', default='SLFN11 (91607)', help='Gene column name')

    args = parser.parse_args()

    try:
        # Load data
        df = pd.read_csv(args.input, usecols=['ModelID', args.gene])

        # Assign quartile labels to ALL models
        # q=4 creates quartiles: [0-25%, 25-50%, 50-75%, 75-100%]
        labels = ['low', 'low-mid', 'high-mid', 'high']
        df['quartile'] = pd.qcut(df[args.gene], q=4, labels=labels)

        # Rename columns to match requirements
        result = df.rename(columns={'ModelID': 'model id', args.gene: 'expression'})

        # Save all rows to CSV
        result[['model id', 'expression', 'quartile']].to_csv(args.output, index=False)

        print(f"Success! Processed {len(result)} models.")
        print(f"Output saved to: {args.output}")

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Column '{args.gene}' or 'ModelID' not found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
