# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This project uses RNNs and Transformers to model and predict NHL game data. It is currently in the initial setup phase — no source code, dependencies, or infrastructure exists yet.

## Planned Architecture

When implementation begins, the expected structure is:
- **Data pipeline:** Fetch and preprocess NHL game data (likely via the NHL Stats API or a third-party wrapper)
- **Models:** RNN-based sequence models and Transformer-based models for game outcome prediction
- **Language/framework:** Python with PyTorch or TensorFlow (TBD)

## Development Setup

No build system, dependencies, or test runner has been configured yet. Once established, update this section with:
- How to create/activate the virtual environment
- How to install dependencies
- How to run training scripts
- How to run tests (e.g., `pytest tests/`)
- How to lint (e.g., `ruff check .` or `flake8`)
