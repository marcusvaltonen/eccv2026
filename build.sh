#!/bin/bash
usage="$(basename "$0") [-h] [-b] -- Build script for ECCV2026

Parameters:
    -h, --help       Show this help message
    -t, --test       Build test scripts"

set -euo pipefail

TEST=OFF
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--test) TEST=ON ;;
        -h|--help) echo "$usage"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

build_dir=_build

# Display information
echo "Configuring build for ECCV2026"
echo "  - Build directory: $build_dir"

# Change directory depending on whether it is debug or release mode
mkdir -p $build_dir && cd $build_dir

cmake -DWITH_TEST=$TEST ..

# Build (for Make on Unix equivalent to `make -j $(nproc)`)
if [[ "$OSTYPE" == "darwin"* ]]; then
  # Mac OSX does not have nproc
  cmake --build . -- -j $(sysctl -n hw.logicalcpu)
else
  cmake --build . -- -j $(nproc)
fi

