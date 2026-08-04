#!/usr/bin/env bash

set -euo pipefail

readonly EXPECTED_PROJECT_NAME="cardiac-mri-3d-segmentation"
readonly CURRENT_PROJECT_NAME="$(basename -- "$PWD")"

if [[ "$CURRENT_PROJECT_NAME" != "$EXPECTED_PROJECT_NAME" ]]; then
    printf \
        'Error: run this script from the "%s" project root.\n' \
        "$EXPECTED_PROJECT_NAME" >&2
    exit 1
fi

readonly -a DIRECTORIES=(
    "configs"
    "data/raw/acdc"
    "docs"
    "notebooks"
    "scripts"
    "src/cardiac_segmentation"
    "src/cardiac_segmentation/config"
    "src/cardiac_segmentation/data"
    "src/cardiac_segmentation/visualization"
    "tests/unit"
    "tests/integration"
    "tests/fixtures"
    "artifacts/dataset_inspection"
)

readonly -a EMPTY_FILES=(
    "configs/data.yaml"
    "docs/dataset.md"
    "scripts/inspect_dataset.py"
    "src/cardiac_segmentation/__init__.py"
    "src/cardiac_segmentation/config/__init__.py"
    "src/cardiac_segmentation/config/schemas.py"
    "src/cardiac_segmentation/config/loader.py"
    "src/cardiac_segmentation/data/__init__.py"
    "src/cardiac_segmentation/data/metadata.py"
    "src/cardiac_segmentation/data/acdc_indexer.py"
    "src/cardiac_segmentation/data/inspection.py"
    "src/cardiac_segmentation/visualization/__init__.py"
    "src/cardiac_segmentation/visualization/slice_visualizer.py"
    "tests/unit/test_acdc_indexer.py"
    "tests/unit/test_metadata.py"
    "tests/unit/test_inspection.py"
    "tests/integration/test_dataset_inspection.py"
    "tests/fixtures/.gitkeep"
    "data/raw/acdc/.gitkeep"
    "artifacts/dataset_inspection/.gitkeep"
    ".gitignore"
    "pyproject.toml"
    "README.md"
    "LICENSE"
)

for directory in "${DIRECTORIES[@]}"; do
    mkdir -p -- "$directory"
done

for file_path in "${EMPTY_FILES[@]}"; do
    if [[ ! -e "$file_path" ]]; then
        : > "$file_path"
    fi
done

readonly NOTEBOOK_PATH="notebooks/acdc_dataset_exploration.ipynb"

if [[ ! -s "$NOTEBOOK_PATH" ]]; then
    cat > "$NOTEBOOK_PATH" <<'JSON'
{
  "cells": [],
  "metadata": {
    "kernelspec": {
      "display_name": "pytorch-env",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
JSON
fi

printf '\nMilestone 1 project structure created successfully.\n\n'

if command -v tree >/dev/null 2>&1; then
    tree -a -L 5 --dirsfirst \
        -I '.git|.mypy_cache|.pytest_cache|.ruff_cache|__pycache__'
else
    find . \
        -path './.git' -prune -o \
        -path '*/__pycache__' -prune -o \
        -print |
        sort
fi
