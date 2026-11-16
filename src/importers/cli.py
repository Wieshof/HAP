"""cli.py - Command-line interface for prediction market data importers.

Usage examples:
  # Import from a local dump file:
  hap-import manifold --dump data/sample_dumps/manifold_sample.json --out data/manifold_exp/
  hap-import metaculus --dump data/sample_dumps/metaculus_sample.json --out data/metaculus_exp/
  hap-import polymarket --dump data/sample_dumps/polymarket_sample.json --out data/polymarket_exp/
  hap-import kalshi --dump data/sample_dumps/kalshi_sample.json --out data/kalshi_exp/

  # Auto-detect platform from dump:
  hap-import auto --dump path/to/dump.json --out data/auto_exp/

  # Fetch live from API:
  hap-import manifold --fetch --limit 50 --out data/manifold_live/ --save-raw raw_manifold.json
  hap-import metaculus --fetch --limit 50 --out data/metaculus_live/ --save-raw raw_metaculus.json

  # Combine dumps from multiple platforms:
  hap-import multi --manifold m.json --metaculus meta.json --out data/combined_exp/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .base import BaseImporter, ImportedDataset, RawMarket
from .kalshi import KalshiImporter
from .manifold import ManifoldImporter
from .metaculus import MetaculusImporter
from .polymarket import PolymarketImporter
from .unified import (
    combine_multiplatform_markets,
    detect_platform_from_file,
    get_importer,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dump",
        type=Path,
        help="Path to local JSON or JSONL dump file",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch live market data from the platform's public API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of markets to fetch when using --fetch (default: 100)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API token (required for Metaculus --fetch; optional elsewhere)",
    )
    parser.add_argument(
        "--save-raw",
        type=Path,
        default=None,
        help="Optional path to save downloaded raw JSON records for offline reuse",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/imported"),
        help="Output directory to write questions.csv and predictions.csv (default: outputs/imported)",
    )
    parser.add_argument(
        "--split",
        choices=["chronological", "random"],
        default="chronological",
        help="Train/test split strategy (default: chronological)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Proportion of questions allocated to train split (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility when using --split random (default: 42)",
    )


def handle_platform_import(
    importer: BaseImporter,
    args: argparse.Namespace,
) -> int:
    """Handle loading, fetching, converting and saving for a single platform."""
    markets: List[RawMarket] = []

    if args.dump:
        print(f"Loading {importer.platform_name} dump from {args.dump}...")
        markets = importer.load_dump_file(args.dump)
        print(f"Parsed {len(markets)} resolved binary markets from {args.dump}")

    elif args.fetch:
        print(f"Fetching up to {args.limit} resolved markets from {importer.platform_name} API...")
        raw_items = importer.fetch_live_raw(limit=args.limit, api_key=args.api_key)
        print(f"Received {len(raw_items)} raw market records from API")

        if args.save_raw:
            args.save_raw.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_raw, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, indent=2)
            print(f"Saved raw JSON records to {args.save_raw}")

        for item in raw_items:
            m = importer.parse_market(item)
            if m is not None:
                markets.append(m)
        print(f"Parsed {len(markets)} valid resolved binary markets")

    else:
        print("Error: Specify either --dump <path> to load local data or --fetch to download from API.", file=sys.stderr)
        return 1

    if not markets:
        print("Error: No valid resolved binary markets could be extracted.", file=sys.stderr)
        return 1

    dataset = importer.to_hap_dataset(
        markets=markets,
        split_strategy=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    q_path, p_path = dataset.write_csv(args.out)
    summary = dataset.summary()

    print(f"\nSuccessfully wrote HAP dataset to {args.out}:")
    print(f"  - Questions: {q_path} ({summary['total_questions']} questions: {summary['train_questions']} train, {summary['test_questions']} test)")
    print(f"  - Predictions: {p_path} ({summary['agent_count']} agents: {', '.join(summary['agents'])})")
    print(f"  - Positive outcome rate: {summary['train_positive_rate']:.1%} train, {summary['test_positive_rate']:.1%} test")
    print(f"\nRun benchmark with:")
    print(f"  hap-experiment --questions {q_path} --predictions {p_path} --out {args.out / 'benchmark'}")
    return 0


def handle_auto_import(args: argparse.Namespace) -> int:
    """Auto-detect platform from dump file."""
    if not args.dump:
        print("Error: --dump <path> is required for auto mode.", file=sys.stderr)
        return 1

    try:
        platform = detect_platform_from_file(args.dump)
        print(f"Detected platform: {platform}")
    except Exception as e:
        print(f"Error detecting platform: {e}", file=sys.stderr)
        return 1

    importer = get_importer(platform)
    return handle_platform_import(importer, args)


def handle_multi_import(args: argparse.Namespace) -> int:
    """Combine multiple dump files into one benchmark dataset."""
    markets_by_platform: Dict[str, List[RawMarket]] = {}

    mappings = [
        ("manifold", args.manifold),
        ("metaculus", args.metaculus),
        ("polymarket", args.polymarket),
        ("kalshi", args.kalshi),
    ]

    for plat, path in mappings:
        if path:
            print(f"Loading {plat} dump from {path}...")
            imp = get_importer(plat)
            mkts = imp.load_dump_file(path)
            print(f"  Loaded {len(mkts)} markets from {plat}")
            markets_by_platform[plat] = mkts

    if not markets_by_platform:
        print("Error: Provide at least one dump file (--manifold, --metaculus, --polymarket, --kalshi).", file=sys.stderr)
        return 1

    dataset = combine_multiplatform_markets(
        markets_by_platform=markets_by_platform,
        split_strategy=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    q_path, p_path = dataset.write_csv(args.out)
    summary = dataset.summary()

    print(f"\nSuccessfully created combined multi-platform dataset in {args.out}:")
    print(f"  - Questions: {q_path} ({summary['total_questions']} total: {summary['train_questions']} train, {summary['test_questions']} test)")
    print(f"  - Predictions: {p_path} ({summary['agent_count']} agents: {', '.join(summary['agents'])})")
    print(f"\nRun benchmark with:")
    print(f"  hap-experiment --questions {q_path} --predictions {p_path} --out {args.out / 'benchmark'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hap-import",
        description="Import resolved prediction market dumps (Manifold, Metaculus, Polymarket, Kalshi) into HAP datasets.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Platform or import mode")

    # Manifold
    p_mnf = subparsers.add_parser("manifold", help="Import from Manifold Markets")
    _add_common_arguments(p_mnf)

    # Metaculus
    p_meta = subparsers.add_parser("metaculus", help="Import from Metaculus")
    _add_common_arguments(p_meta)

    # Polymarket
    p_poly = subparsers.add_parser("polymarket", help="Import from Polymarket")
    _add_common_arguments(p_poly)

    # Kalshi
    p_kalshi = subparsers.add_parser("kalshi", help="Import from Kalshi")
    _add_common_arguments(p_kalshi)

    # Auto
    p_auto = subparsers.add_parser("auto", help="Auto-detect platform from dump file")
    _add_common_arguments(p_auto)

    # Multi
    p_multi = subparsers.add_parser("multi", help="Combine dumps from multiple platforms")
    p_multi.add_argument("--manifold", type=Path, help="Path to Manifold Markets dump")
    p_multi.add_argument("--metaculus", type=Path, help="Path to Metaculus dump")
    p_multi.add_argument("--polymarket", type=Path, help="Path to Polymarket dump")
    p_multi.add_argument("--kalshi", type=Path, help="Path to Kalshi dump")
    p_multi.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/combined_imported"),
        help="Output directory (default: outputs/combined_imported)",
    )
    p_multi.add_argument(
        "--split",
        choices=["chronological", "random"],
        default="chronological",
        help="Train/test split strategy (default: chronological)",
    )
    p_multi.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Train split proportion (default: 0.5)",
    )
    p_multi.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    try:
        return _dispatch(args)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.subcommand == "metaculus" and getattr(args, "fetch", False) and not args.api_key:
        print(
            "Error: the Metaculus API only answers signed-in users. Pass your "
            "token with --api-key, or import a saved dump with --dump.",
            file=sys.stderr,
        )
        return 1
    if args.subcommand == "manifold":
        return handle_platform_import(ManifoldImporter(), args)
    elif args.subcommand == "metaculus":
        return handle_platform_import(MetaculusImporter(), args)
    elif args.subcommand == "polymarket":
        return handle_platform_import(PolymarketImporter(), args)
    elif args.subcommand == "kalshi":
        return handle_platform_import(KalshiImporter(), args)
    elif args.subcommand == "auto":
        return handle_auto_import(args)
    elif args.subcommand == "multi":
        return handle_multi_import(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
