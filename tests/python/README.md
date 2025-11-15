# Python Tests for AMSwarm

This directory contains Python tests for the AMSwarm project.

## Running Tests

To run the Python tests, first install the test dependencies:

```bash
pip install pytest
```

Or install with the test extras:

```bash
pip install -e .[test]
```

Then run the tests:

```bash
pytest tests/python
```

## Test Structure

- `test_import.py` - Basic tests for importing the AMSwarm Python bindings
  - `test_import_amswarm()` - Tests that the amswarm module can be imported (skips if not installed)
  - `test_dummy_pass()` - A dummy test that always passes
  - `test_basic_arithmetic()` - Basic test to verify pytest is working

## CI Integration

Python tests are automatically run in CI via the `python-tests` job in `.github/workflows/ci.yml`.
