import argparse
from pathlib import Path

from .config import ExperimentConfig
from .pipeline import prepare_dataset, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="f1-pit-strategy"); commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare"); prepare.add_argument("--data-dir", type=Path, required=True); prepare.add_argument("--output-dir", type=Path, default=Path("outputs/prepared"))
    run = commands.add_parser("run"); run.add_argument("--data-dir", type=Path, required=True); run.add_argument("--output-dir", type=Path, default=Path("outputs")); run.add_argument("--oracle-unlabeled", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": prepare_dataset(args.data_dir, args.output_dir)
    else: print(run_pipeline(ExperimentConfig(args.data_dir, args.output_dir, oracle_unlabeled=args.oracle_unlabeled)).to_string(index=False))


if __name__ == "__main__": main()
