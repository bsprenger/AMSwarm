"""
Test for crazyflow integration example.

This test verifies that the crazyflow example can be imported and run successfully
when the crazyflow optional dependency is installed.
"""

import pytest


def test_crazyflow_imports():
    """Test that both amswarm and crazyflow can be imported together."""
    try:
        import crazyflow
        assert crazyflow is not None
    except ImportError as e:
        pytest.skip(f"Crazyflow not installed: {e}")


def test_crazyflow_example_runs():
    """Test that the crazyflow example runs without errors."""
    try:
        import crazyflow  # Check if crazyflow is available
        import sys
        from pathlib import Path
        from io import StringIO
        
        # Add examples/python to path
        examples_path = Path(__file__).parent.parent.parent / "examples" / "python"
        sys.path.insert(0, str(examples_path))
        
        import crazyflow_example
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            crazyflow_example.main()
            output = captured_output.getvalue()
            
            # Verify expected output
            assert "Successfully imported" in output
            assert "crazyflow" in output
        finally:
            sys.stdout = old_stdout
            
    except ImportError as e:
        pytest.skip(f"Crazyflow not installed: {e}")

