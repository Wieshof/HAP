# Earlier HAP design (September 2026)

These files are the original HAP design. They are kept for reference and are not the current focus. The current first step is the diversity benchmark described in the main README.

The main README is self-contained: it explains the essay's ideas and the auction in plain language. You do not need these files to understand or run the benchmark.

- `machine_and_market_revised.md`: the original essay, *The Machine and the Market*. It covers Hayek's dispersed knowledge, stigmergy, Ostrom-style governance, Hirschman's exit and voice, the reverse Vickrey auction and the entropy check against herding.
- `hap_core.py`: a reference auction engine. It binds agent IDs to Ed25519 keys, accepts one signed bid per agent before a deadline and within a maximum price, and pays the winner the second-lowest bid, or the maximum price when only one bid arrives. It also includes a semantic-entropy monitor.
- `hap_gateway.py`: a small FastAPI service over that engine.

## Status

This is research code, not production software. It does not provide Sybil resistance, payment, persistence or quality checks. The essay and the code both say so.

The demo needs Python 3.10+ and the `cryptography` package; the gateway also needs `fastapi` and `pydantic`.

```bash
pip install cryptography
python hap_core.py
```

The demo ran successfully on 25 October 2026. It prints the auction winner and payout, then entropy values for a diverse group and a herding group.
