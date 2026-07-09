"""deeplob-serve: deep learning on limit order books, served from a C++ inference engine.

This module currently proves the toolchain (uv, ruff, mypy strict, pytest, CI) works end to
end before any real module exists -- mirrors liquibook-x's own smoke/ pattern. Real modules
(data/, models/, training/, evaluation/, export/, simulation/) replace this milestone by
milestone.
"""

__version__ = "0.1.0-m1"
