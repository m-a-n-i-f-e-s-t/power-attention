# Get version from pyproject.toml
VERSION := $(shell python scripts/get_version.py)
PACKAGE_NAME := power-attention
# Find Python 3.11+
PYTHON := $(shell for py in python3.12 python3.11 python3 python; do \
    if command -v $$py >/dev/null && $$py --version 2>&1 | grep -q "Python 3.1[1-9]"; then \
        echo $$py; \
        break; \
    fi \
done)

ifeq ($(PYTHON),)
    $(error Python 3.11 or higher is required. Please install Python 3.11+)
endif

.PHONY: clean check-version check-test-version release release-test help

# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Release commands
release: clean check-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m twine check dist/*
	@echo "Uploading to PyPI..."
	python -m twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean check-test-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m twine check dist/*
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Help
help:
	@echo "Available commands:"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make release       - Release to PyPI (includes version check)"
	@echo "  make release-test  - Release to TestPyPI"
	@echo "  make check-version - Check version against PyPI"
	@echo "  make check-test-version - Check version against TestPyPI"
	@echo "Current version: $(VERSION)" 
