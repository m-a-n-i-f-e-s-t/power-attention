# Get version from pyproject.toml
VERSION := $(shell python scripts/get_version.py)
PACKAGE_NAME := power-attention
PYTHON := python3

# Allow overriding venv path through environment variable, default to .venv
VENV_DIR ?= $(if $(POWER_ATTENTION_VENV_PATH),$(POWER_ATTENTION_VENV_PATH),.venv)
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest

.PHONY: all build test benchmark release release-test clean check-version venv deps deps-dev deps-train

all: build test

dev: venv
	CC=gcc CXX=g++ $(PIP) install -e .

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

deps: venv
	$(PIP) install .

deps-dev: deps
	$(PIP) install ".[dev]"

deps-train: deps
	$(PIP) install ".[train]"

# Development commands
test: deps-dev
	$(PYTEST) power_attention/tests.py -v

benchmark: deps-dev
	$(VENV_DIR)/bin/python test/benchmark.py

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@$(VENV_DIR)/bin/python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@$(VENV_DIR)/bin/python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/

# Release commands
release: clean deps-dev check-version
	@echo "Building wheels with cibuildwheel..."
	$(VENV_DIR)/bin/python -m cibuildwheel --output-dir dist
	$(VENV_DIR)/bin/twine check dist/*
	@echo "Uploading to PyPI..."
	$(VENV_DIR)/bin/twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean deps-dev check-test-version
	@echo "Building wheels with cibuildwheel..."
	$(VENV_DIR)/bin/python -m cibuildwheel --output-dir dist
	$(VENV_DIR)/bin/twine check dist/*
	@echo "Uploading to TestPyPI..."
	$(VENV_DIR)/bin/twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Help
help:
	@echo "Available commands:"
	@echo "Environment variables:"
	@echo "  POWER_ATTENTION_VENV_PATH - Override default virtualenv location (.venv)"
	@echo ""
	@echo "Commands:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make deps          - Install base dependencies"
	@echo "  make deps-dev      - Install development dependencies"
	@echo "  make deps-train    - Install training dependencies"
	@echo "  make test          - Run tests"
	@echo "  make benchmark     - Run benchmarks"
	@echo "  make build         - Build package"
	@echo "  make fast          - Quick build for development"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make release       - Release to PyPI (includes version check)"
	@echo "  make release-test  - Release to TestPyPI"
	@echo "  make check-version - Check version against PyPI"
	@echo "  make check-test-version - Check version against TestPyPI"
	@echo "Current version: $(VERSION)" 
