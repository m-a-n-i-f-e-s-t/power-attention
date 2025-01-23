# Release Process

This guide explains how to create and publish new releases of Power Attention.

<div className="bg-purple-50 border-l-4 border-purple-500 p-4 my-6">
  <p className="text-purple-700">
    <strong>Note:</strong> Only maintainers with proper permissions can create official releases.
  </p>
</div>

## Version Scheme

We follow semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

## Release Steps

1. Update version in `pyproject.toml`:
```toml
[project]
name = "power-attention"
version = "1.2.3"  # New version number
```

2. Check version against PyPI:
```bash
make check-version
```

3. Run test suite:
```bash
pytest
```

4. Build and release on TestPyPI:
```bash
make release-test
```

5. If successful, build and release to PyPI:
```bash
make release
```

<div className="bg-blue-50 border-l-4 border-blue-500 p-4 my-6">
  <p className="text-blue-700">
    <strong>Tip:</strong> Always test releases on TestPyPI first to catch any packaging issues.
  </p>
</div>

## Release Checklist

Before releasing:

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated
- [ ] Version (pyproject.toml) is bumped
- [ ] Release notes are prepared
- [ ] TestPyPI release works
- [ ] Git tag is created

## After Release

1. Create a GitHub release with release notes
2. Announce in relevant channels
3. Update documentation site if needed
4. Monitor issues for any release-related problems

## Troubleshooting

If the release fails:

1. Check PyPI/TestPyPI credentials
2. Verify version number is unique
3. Ensure all files are included in the package
4. Check build artifacts for issues

For help with releases, contact the maintainers team. 