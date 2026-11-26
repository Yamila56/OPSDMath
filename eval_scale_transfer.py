"""Compatibility entry point for the packaged implementation."""

import sys

from opsdmath.evaluation import budget as implementation


if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
