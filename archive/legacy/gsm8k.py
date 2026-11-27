"""Compatibility entry point for the packaged implementation."""

import sys

from opsdmath.data import gsm8k as implementation


if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
