
# AI Guide

## Test Guide

- Use only `pytest` (no `unittest`, no test classes).
- All tests are top-level functions named `test_*`.
- Use `@pytest.fixture` for setup/data.
- Use plain `assert` for checks.
- Use `@pytest.mark.parametrize` for multiple cases.
- Use `pytest.raises` for exceptions.
- Use `tmp_path`/`tmpdir` for temp files.
- Each test must be independent.

Example:
```python
import pytest
from mymodule import myfunc

@pytest.fixture
def sample_data():
    return [1, 2, 3]

def test_myfunc_basic(sample_data):
    assert myfunc(sample_data) == expected_result

@pytest.mark.parametrize("input,expected", [(1, 2), (2, 3)])
def test_myfunc_param(input, expected):
    assert myfunc(input) == expected

def test_myfunc_error():
    with pytest.raises(ValueError):
        myfunc(bad_input)
```

## Docstring Style Guide (PEP287 reST)

Use short, concise reST (PEP287) docstrings for all modules, classes, and functions.

### Function Docstring Template

```python
def func(arg1, arg2):
    """Short summary.

    :param int arg1: Description
    :param str arg2: Description
    :returns bool: Description
    :raises ValueError: If something goes wrong
    """
```

### Class Docstring Template

```python
class MyClass:
    """Short summary.

    :ivar int attr: Description
    :ivar str other: Description
    """
```

### Module Docstring Template

```python
"""Short summary of the module.

:var str VERSION: Module version
"""
```

### Best Practices

- Use a single short summary line if possible.
- Use field lists with type on the same line as param/return (e.g., :param int foo: ...)
- Only document what isn’t obvious from the signature.
- Use imperative mood for descriptions.
- Prefer single-line docstrings for trivial functions.

### Examples

```python
def add(a, b):
    """Add two numbers.

    :param float a: First number
    :param float b: Second number
    :returns float: Sum of a and b
    """

class User:
    """User object.

    :ivar str name: User name
    :ivar int age: User age
    """
```
