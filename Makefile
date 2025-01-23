# Get version from pyproject.toml
VERSION := $(shell python scripts/get_version.py)
PACKAGE_NAME := power-attention

.PHONY: clean check-version check-test-version release release-test help deps-dev kernel refresh-deps refresh-deps-dev

PIP := pip
PYTEST := pytest
PYTHON := python

define install_group_deps
	$(PYTHON) -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["dependency-groups"]["$(1)"]))' | $(PIP) install -r /dev/stdin
endef

define install_deps
	$(PYTHON) -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]))' | $(PIP) install -r /dev/stdin
endef

define install_group_deps
	$(PYTHON) -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["dependency-groups"]["$(1)"]))' | $(PIP) install -r /dev/stdin
endef

define uninstall_deps
	$(PYTHON) -c 'import tomllib; deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]; [print(dep.split(">=")[0]) for dep in deps]' | xargs -n 1 $(PIP) uninstall -y	
endef

kernel:
	@python setup.py build_ext --inplace

deps-dev:
	$(call install_group_deps,dev)

refresh-deps:
	@echo "Uninstalling dependencies..."
	$(call uninstall_deps)
	@echo "Reinstalling dependencies..."
	$(call install_deps)

refresh-deps-dev:
	@echo "Uninstalling development dependencies..."
	$(call uninstall_dev_deps)
	@echo "Reinstalling development dependencies..."
	$(call install_dev_deps)

# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/ $(VENV_DIR)/.deps_*

kernel:
	@python setup.py build_ext --inplace

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Release commands
release:
	python -m twine check dist/*
	@echo "Uploading to PyPI..."
	python -m twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean check-test-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m build -s
	python -m twine check dist/*
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Visualization
plot-regressions:
	@echo "Generating regression visualization..."
	$(PYTHON) perf/plot_regressions.py

# Help
help:
	@echo "Available commands:"
	@echo "  make kernel             - Build kernel and (re)install it"
	@echo "  make deps               - Install required dependencies"
	@echo "  make deps-dev           - Install development dependencies"
	@echo "  make refresh-deps       - Refresh required dependencies"
	@echo "  make refresh-deps-dev   - Refresh development dependencies"
	@echo "  make clean              - Clean build artifacts"
	@echo "  make release            - Release to PyPI (includes version check)"
	@echo "  make release-test       - Release to TestPyPI"
	@echo "  make check-version      - Check version against PyPI"
	@echo "  make check-test-version - Check version against TestPyPI"
	@echo "  make plot-regressions   - Generate interactive regression visualization"
	@echo "Current version: $(VERSION)" 
