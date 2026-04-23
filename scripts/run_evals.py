#!/usr/bin/env python3
"""Run evaluation tests via pytest, logging results to MLflow."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation tests")
    parser.add_argument("--run-name", help="MLflow run name for this eval session")
    args, extra_args = parser.parse_known_args()

    cmd = [sys.executable, "-m", "pytest", "-s", "-m", "deepeval"] + extra_args
    if args.run_name:
        cmd += ["--run-name", args.run_name]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
