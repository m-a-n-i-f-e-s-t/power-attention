# Contributing Guidelines

We welcome contributions to Power Attention! This guide will help you get started with development.

<div className="bg-green-50 border-l-4 border-green-500 p-4 my-6">
  <p className="text-green-700">
    <strong>Good First Issues:</strong> Look for issues tagged with `good-first-issue` in our GitHub repository if you're just getting started.
  </p>
</div>

## Development Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/yourusername/power-attention.git
cd power-attention
```

2. Set up the development environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

3. Create a new branch for your feature/fix:
```bash
git checkout -b feature-name
```

## Development Guidelines

### Code Style

- Python code should follow PEP 8
- CUDA code should follow the existing style in the codebase
- Use meaningful variable names and add comments for complex logic
- Keep functions focused and modular

### Documentation

- Add docstrings to new functions and classes
- Update relevant documentation files in `docs/`
- Include examples for new features
- Keep the README up to date

### Testing

- Add tests for new features
- Ensure all tests pass: `make test`
- Include both unit tests and integration tests where appropriate
- For CUDA kernels, add precision tests comparing against reference implementations

### Performance

- For CUDA kernels, include benchmarks showing performance impact
- Compare against baseline implementations where relevant
- Consider memory usage and optimization opportunities
- Test with different batch sizes and sequence lengths

## Pull Request Process

1. Ensure your code follows our style guidelines
2. Add or update tests as needed
3. Update documentation for any new features
4. Run the test suite: `make test`
5. Run benchmarks if performance-critical code was changed: `make benchmark`
6. Create a Pull Request with a clear description of changes
7. Wait for review and address any feedback

<div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 my-6">
  <p className="text-yellow-700">
    <strong>Note:</strong> Before submitting a large PR, it's recommended to open an issue first to discuss the proposed changes.
  </p>
</div>

## Getting Help

- Check existing issues and discussions on GitHub
- Join our community chat (if available)
- Feel free to ask questions in issues or discussions
- Tag maintainers if you need specific help 