"""Usage: uv run python scripts/add_ar_reference.py outputs/OLD outputs/NEW"""
import argparse
from pathlib import Path
from gold_forecast.reference import add_ar_reference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(add_ar_reference(args.source, args.destination))
