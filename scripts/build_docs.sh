#!/bin/bash
set -eu

cd "$(dirname $0)/.."
make -C docs html

