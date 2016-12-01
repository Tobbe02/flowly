#!/bin/bash
set -eu

cd "$(dirname $0)/.."

mkdir -p tmp/coverage

echo ":: check code style"
python -m flake8 tests/ flowly/

echo ":: run tests"
python -m coverage run -m pytest tests

echo ":: generate coverage report"
python -m coverage html -d tmp/coverage

echo ":: done"

