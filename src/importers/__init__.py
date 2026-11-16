"""importers - Prediction market data loaders for HAP.

Supports resolved binary dumps and live API endpoints for:
  * Manifold Markets (https://docs.manifold.markets/api)
  * Metaculus (https://www.metaculus.com/api/)
  * Polymarket (https://data-api.polymarket.com/v2/docs#description/introduction)
  * Kalshi (https://docs.kalshi.com/welcome)
"""
from .base import (
    AgentDefinition,
    BaseImporter,
    ImportedDataset,
    RawMarket,
    sanitize_probability,
    split_chronological,
    split_random,
)
from .kalshi import KalshiImporter
from .manifold import ManifoldImporter
from .metaculus import MetaculusImporter
from .polymarket import PolymarketImporter
from .unified import (
    combine_multiplatform_markets,
    detect_platform_from_file,
    get_importer,
    load_dump_file,
)

__all__ = [
    "AgentDefinition",
    "BaseImporter",
    "ImportedDataset",
    "RawMarket",
    "sanitize_probability",
    "split_chronological",
    "split_random",
    "ManifoldImporter",
    "MetaculusImporter",
    "PolymarketImporter",
    "KalshiImporter",
    "get_importer",
    "detect_platform_from_file",
    "load_dump_file",
    "combine_multiplatform_markets",
]
