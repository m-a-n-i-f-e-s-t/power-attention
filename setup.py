import os
import re
import tomllib
from setuptools import find_packages, setup

this_dir = os.path.dirname(os.path.abspath(__file__))

def get_version():
    """Get version from pyproject.toml."""
    with open(os.path.join(this_dir, "pyproject.toml")) as f:
        content = f.read()
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return version_match.group(1)

def get_dependencies():
    """Get dependencies from pyproject.toml."""
    with open(os.path.join(this_dir, "pyproject.toml"), "rb") as f:
        pyproject = tomllib.load(f)
    
    # Get main dependencies
    deps = pyproject["project"]["dependencies"]
    
    # Add dev dependencies
    groups = pyproject.get("project.optional-dependencies", {})
    deps.extend(groups.get("dev", []))
    
    return list(set(deps))  # Remove duplicates


# Development dependencies
dev_requires = get_dependencies()

setup(
    name="power-attention",
    version=get_version(),
    packages=find_packages(
        exclude=('build', 'csrc', 'include', 'tests', 'dist', 'benchmarks'),
    ),
    extras_require={
        'dev': dev_requires,
    },
)
