"""Command-line interface for AutoPrepML"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from .core import AutoPrepML


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='AutoPrepML - Automated Data Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cleaning
  autoprepml --input data.csv --output cleaned.csv
  
  # Classification task with target column
  autoprepml --input train.csv --output clean_train.csv --task classification --target label
  
  # Generate HTML report
  autoprepml --input data.csv --output cleaned.csv --report report.html
  
  # Use custom config
  autoprepml --input data.csv --output cleaned.csv --config config.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output cleaned CSV file path'
    )
    
    parser.add_argument(
        '--report', '-r',
        help='Output report file path (.html or .json)'
    )
    
    parser.add_argument(
        '--task',
        choices=['classification', 'regression'],
        help='ML task type (affects preprocessing strategy)'
    )
    
    parser.add_argument(
        '--target',
        help='Name of target column (for classification/regression tasks)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to YAML/JSON configuration file'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation in reports'
    )
    
    parser.add_argument(
        '--detect-only',
        action='store_true',
        help='Only run detection, do not clean data'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if input_path.suffix.lower() != '.csv':
        print("❌ Error: Input file must be a CSV file", file=sys.stderr)
        sys.exit(1)
    
    # Validate report format
    if args.report:
        report_path = Path(args.report)
        if report_path.suffix.lower() not in ['.html', '.json']:
            print("❌ Error: Report file must be .html or .json", file=sys.stderr)
            sys.exit(1)
    
    # Load data
    try:
        print(f"📂 Loading data from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize AutoPrepML
    try:
        prep = AutoPrepML(df, config_path=args.config)
        
        if args.no_plots:
            prep.config['reporting']['include_plots'] = False
        
        # Detection phase
        print("\n🔍 Running detection...")
        detection_results = prep.detect(target_col=args.target)
        
        # Print detection summary
        missing_count = len(detection_results.get('missing_values', {}))
        outlier_count = detection_results.get('outliers', {}).get('outlier_count', 0)
        
        print(f"   • Missing values: {missing_count} columns affected")
        print(f"   • Outliers detected: {outlier_count} rows")
        
        if args.target and 'class_imbalance' in detection_results:
            imbalance = detection_results['class_imbalance']
            status = "⚠ Imbalanced" if imbalance['is_imbalanced'] else "✓ Balanced"
            print(f"   • Class distribution: {status}")
        
        if args.detect_only:
            print("\n✅ Detection complete (--detect-only mode)")
            if args.report:
                prep.save_report(args.report)
                print(f"📄 Report saved to {args.report}")
            sys.exit(0)
        
        # Cleaning phase
        print("\n🧹 Cleaning data...")
        clean_df, report = prep.clean(task=args.task, target_col=args.target)
        
        # Save cleaned data
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(args.output, index=False)
        print(f"✅ Cleaned data saved to {args.output}")
        print(f"   Shape: {clean_df.shape[0]} rows × {clean_df.shape[1]} columns")
        
        # Save report
        if args.report:
            prep.save_report(args.report)
            print(f"📄 Report saved to {args.report}")
        
        print("\n🎉 AutoPrepML completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
