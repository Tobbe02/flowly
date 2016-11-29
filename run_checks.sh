#!/bin/bash
set -eu

mkdir -p tmp/coverage
python -m flake8 tests/ flowly/
python -m coverage run -m pytest tests
python -m coverage html -d tmp/coverage
