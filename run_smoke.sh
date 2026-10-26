#!/usr/bin/env sh
# Regenerate the synthetic fixture, run the benchmark, the contribution
# analysis and the buyer test, then the tests.
# Needs Python 3.10+ and nothing else.
set -e
cd "$(dirname "$0")"
python3 src/hap_experiment.py --generate-smoke
python3 src/hap_contribution.py
python3 -m unittest discover -s tests
