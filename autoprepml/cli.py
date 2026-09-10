"""Command-line interface for AutoPrepML"""

import argparse
import json
import sys
import pandas as pd
from pathlib import Path

from . import __version__
from .config import load_config
from .data_plan import DataPlan
from .core import AutoPrepML
from .storage import LocalStorageAdapter


MODERN_COMMANDS = {"inspect", "fit", "validate", "transform", "version"}


def _modern_parser() -> argparse.ArgumentParser:
    """Build the fitted DataPlan command-line interface."""
    parser = argparse.ArgumentParser(
        prog="autoprepml",
        description="Inspect, fit, validate, and transform ML-ready datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Inspect a dataset without fitting transforms")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--target")
    inspect.add_argument("--task", choices=["classification", "regression"])
    inspect.add_argument(
        "--fingerprint-mode", choices=["schema", "sampled", "full"], default="full"
    )

    fit = commands.add_parser("fit", help="Fit and save a reusable DataPlan artifact")
    fit.add_argument("input", type=Path)
    fit.add_argument("--target", required=True)
    fit.add_argument("--task", choices=["classification", "regression"])
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--config", type=Path)
    fit.add_argument("--fingerprint-mode", choices=["schema", "sampled", "full"], default="full")
    fit.add_argument("--fingerprint-sample-rows", type=int, default=1000)
    fit.add_argument("--output-format", choices=["pandas", "sparse", "numpy"], default="pandas")
    fit.add_argument("--random-state", type=int, default=42)

    validate = commands.add_parser("validate", help="Validate data against a saved DataPlan")
    validate.add_argument("input", type=Path)
    validate.add_argument("--plan", required=True, type=Path)
    validate.add_argument("--strict", action="store_true")

    transform = commands.add_parser("transform", help="Transform data with a saved DataPlan")
    transform.add_argument("input", type=Path)
    transform.add_argument("--plan", required=True, type=Path)
    transform.add_argument("--output", required=True, type=Path)

    commands.add_parser("version", help="Print the installed AutoPrepML version")
    return parser


def _read_modern_input(path: Path) -> pd.DataFrame:
    """Read a supported local tabular input for modern commands."""
    return LocalStorageAdapter().read(path)


def _run_modern(argv) -> int:
    """Run a fitted DataPlan command and return a process status."""
    parser = _modern_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    try:
        if args.command == "inspect":
            plan = DataPlan.infer(
                _read_modern_input(args.input),
                target=args.target,
                task=args.task,
                fingerprint_mode=args.fingerprint_mode,
            )
            print(json.dumps(plan.inspect(), indent=2, default=str))
            return 0
        if args.command == "fit":
            frame = _read_modern_input(args.input)
            config = load_config(str(args.config)) if args.config else None
            plan = DataPlan.infer(
                frame,
                target=args.target,
                task=args.task,
                config=config,
                random_state=args.random_state,
                fingerprint_mode=args.fingerprint_mode,
                fingerprint_sample_rows=args.fingerprint_sample_rows,
                output_format=args.output_format,
            ).fit(frame)
            plan.save(args.output)
            print(json.dumps(plan.manifest(), indent=2, default=str))
            return 0
        plan = DataPlan.load(args.plan)
        frame = _read_modern_input(args.input)
        if args.command == "validate":
            report = plan.validate(frame, mode="strict" if args.strict else "compatible")
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0 if report.compatible else 1
        if args.command == "transform":
            transformed = plan.transform(frame)
            if not isinstance(transformed, pd.DataFrame):
                raise ValueError("transform command requires a plan with pandas or sparse output")
            LocalStorageAdapter().write(transformed, args.output)
            print(f"Transformed data written to {args.output}")
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unsupported command: {args.command}")
    return 2


def main(argv=None):
    """Main CLI entrypoint."""
    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args and command_args[0] in MODERN_COMMANDS:
        return _run_modern(command_args)
    parser = argparse.ArgumentParser(
        description="AutoPrepML - Automated Data Preprocessing Pipeline",
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
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")

    parser.add_argument("--output", "-o", help="Output cleaned CSV file path")

    parser.add_argument("--report", "-r", help="Output report file path (.html or .json)")

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="ML task type (affects preprocessing strategy)",
    )

    parser.add_argument(
        "--target", help="Name of target column (for classification/regression tasks)"
    )

    parser.add_argument("--config", "-c", help="Path to YAML/JSON configuration file")

    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot generation in reports"
    )

    parser.add_argument(
        "--detect-only", action="store_true", help="Only run detection, do not clean data"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    if not args.detect_only and not args.output:
        parser.error("--output is required unless --detect-only is used")

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 2

    if input_path.suffix.lower() != ".csv":
        print("Error: Input file must be a CSV file", file=sys.stderr)
        return 2

    # Validate report format
    if args.report:
        report_path = Path(args.report)
        if report_path.suffix.lower() not in [".html", ".json"]:
            print("Error: Report file must be .html or .json", file=sys.stderr)
            return 2

    # Load data
    try:
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        return 1

    # Initialize AutoPrepML
    try:
        prep = AutoPrepML(df, config_path=args.config)

        if args.no_plots:
            prep.config["reporting"]["include_plots"] = False

        # Detection phase
        print("\nRunning detection...")
        detection_results = prep.detect(target_col=args.target)

        # Print detection summary
        missing_count = len(detection_results.get("missing_values", {}))
        outlier_count = detection_results.get("outliers", {}).get("outlier_count", 0)

        print(f"   Missing values: {missing_count} columns affected")
        print(f"   Outliers detected: {outlier_count} rows")

        if args.target and "class_imbalance" in detection_results:
            imbalance = detection_results["class_imbalance"]
            status = "Imbalanced" if imbalance["is_imbalanced"] else "Balanced"
            print(f"   Class distribution: {status}")

        if args.detect_only:
            print("\nDetection complete (--detect-only mode)")
            if args.report:
                prep.save_report(args.report)
                print(f"Report saved to {args.report}")
            return 0

        # Cleaning phase
        print("\nCleaning data...")
        clean_df, report = prep.clean(task=args.task, target_col=args.target)

        # Save cleaned data
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(args.output, index=False)
        print(f"Cleaned data saved to {args.output}")
        print(f"   Shape: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")

        # Save report
        if args.report:
            prep.save_report(args.report)
            print(f"Report saved to {args.report}")

        print("\nAutoPrepML completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
